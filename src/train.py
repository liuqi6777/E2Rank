import logging
import json
import os
import pathlib
import sys

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import HfArgumentParser, Trainer as HFTrainer, set_seed

from config import DataArguments, LoraArguments, ModelArguments, RLArguments, TrainingArguments
from grpo import GRPOModel
from ranking_data import RankingDataCollator, RankingDataset


logger = logging.getLogger(__name__)


def get_deepspeed_zero_stage(deepspeed_config) -> int | None:
    if deepspeed_config is None:
        return None

    if isinstance(deepspeed_config, dict):
        config = deepspeed_config
    elif isinstance(deepspeed_config, str):
        if os.path.isfile(deepspeed_config):
            with open(deepspeed_config, "r", encoding="utf-8") as fp:
                config = json.load(fp)
        else:
            try:
                config = json.loads(deepspeed_config)
            except json.JSONDecodeError:
                return None
    else:
        return None

    zero_optimization = config.get("zero_optimization")
    if not isinstance(zero_optimization, dict):
        return None

    stage = zero_optimization.get("stage")
    return int(stage) if stage is not None else None


def resolve_gradient_checkpointing_kwargs(training_args: TrainingArguments, lora_args: LoraArguments) -> dict:
    gradient_checkpointing_kwargs = dict(training_args.gradient_checkpointing_kwargs or {})
    zero_stage = get_deepspeed_zero_stage(training_args.deepspeed)

    if lora_args.lora_enabled and zero_stage == 3:
        if gradient_checkpointing_kwargs.get("use_reentrant") is not True:
            logger.warning(
                "Detected DeepSpeed ZeRO-3 with LoRA and gradient checkpointing; "
                "overriding use_reentrant=True because use_reentrant=False can trigger "
                "torch.utils.checkpoint.CheckpointError with empty ZeRO-3 parameter shards."
            )
        gradient_checkpointing_kwargs["use_reentrant"] = True
    else:
        gradient_checkpointing_kwargs.setdefault("use_reentrant", False)

    return gradient_checkpointing_kwargs


def save_model_for_trainer(trainer: HFTrainer, output_dir: str) -> None:
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def main() -> None:
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments, RLArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, lora_args, rl_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, lora_args, rl_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("RL parameters %s", rl_args)

    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        trust_remote_code=True,
        cache_dir=model_args.cache_dir,
    )
    backbone = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        trust_remote_code=True,
        # attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        padding_side="left",
        cache_dir=model_args.cache_dir,
        trust_remote_code=True,
    )

    if lora_args.lora_enabled:
        if lora_args.lora_path:
            print(f"Loading LoRA from {lora_args.lora_path}")
            backbone = PeftModel.from_pretrained(
                backbone,
                lora_args.lora_path,
                is_trainable=True,
            )
        else:
            print("Initializing LoRA")
            lora_config = LoraConfig(
                r=lora_args.lora_r,
                lora_alpha=lora_args.lora_alpha,
                target_modules=lora_args.lora_target_modules,
                lora_dropout=lora_args.lora_dropout,
                bias=lora_args.lora_bias,
                task_type="FEATURE_EXTRACTION",
            )
            backbone = get_peft_model(backbone, lora_config)
        backbone.print_trainable_parameters()

    model = GRPOModel(
        model=backbone,
        rl_mode=rl_args.rl_mode,
        group_size=rl_args.group_size,
        sigma=rl_args.sigma,
        sigma_learnable=rl_args.sigma_learnable,
        query_reward_type=rl_args.query_reward_type,
        listwise_reward_type=rl_args.listwise_reward_type,
        query_reward_ndcg_k=rl_args.query_reward_ndcg_k,
        listwise_reward_ndcg_k=rl_args.listwise_reward_ndcg_k,
        query_mixed_contrastive_weight=rl_args.query_mixed_contrastive_weight,
        query_mixed_ndcg_weight=rl_args.query_mixed_ndcg_weight,
        listwise_mixed_contrastive_weight=rl_args.listwise_mixed_contrastive_weight,
        listwise_mixed_ndcg_weight=rl_args.listwise_mixed_ndcg_weight,
        query_contrastive_use_in_batch_negatives=rl_args.query_contrastive_use_in_batch_negatives,
        listwise_contrastive_use_in_batch_negatives=rl_args.listwise_contrastive_use_in_batch_negatives,
        listwise_loss_weight=rl_args.listwise_loss_weight,
        advantage_norm=rl_args.advantage_norm,
        query_relevance_scheme=rl_args.query_relevance_scheme,
        listwise_relevance_scheme=rl_args.listwise_relevance_scheme,
    )
    model.train()

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = resolve_gradient_checkpointing_kwargs(
            training_args=training_args,
            lora_args=lora_args,
        )
        if training_args.gradient_checkpointing_kwargs.get("use_reentrant", True):
            model.enable_input_require_grads()
        logger.info("Gradient checkpointing kwargs %s", training_args.gradient_checkpointing_kwargs)

    class Trainer(HFTrainer):
        train_metric_names = (
            "query_loss",
            "listwise_loss",
            "query_reward",
            "listwise_reward",
            "query_sigma",
            "listwise_sigma",
        )

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._train_metric_sums: dict[str, torch.Tensor] = {}
            self._train_metric_updates = 0

        def _accumulate_train_metrics(self, outputs) -> None:
            for metric_name in self.train_metric_names:
                if isinstance(outputs, dict):
                    metric_value = outputs.get(metric_name)
                else:
                    metric_value = getattr(outputs, metric_name, None)
                if metric_value is None:
                    continue
                if not isinstance(metric_value, torch.Tensor):
                    metric_value = torch.tensor(metric_value, device=self.args.device, dtype=torch.float32)
                metric_value = metric_value.detach()
                if metric_value.numel() != 1:
                    metric_value = metric_value.mean()
                metric_value = metric_value.to(device=self.args.device, dtype=torch.float32)
                self._train_metric_sums[metric_name] = self._train_metric_sums.get(
                    metric_name,
                    torch.zeros((), device=self.args.device, dtype=torch.float32),
                ) + metric_value
            self._train_metric_updates += 1

        def _consume_train_metrics(self) -> dict[str, float]:
            if self._train_metric_updates == 0:
                return {}

            logs = {}
            metric_count = torch.tensor(
                float(self._train_metric_updates),
                device=self.args.device,
                dtype=torch.float32,
            )
            total_metric_count = self._nested_gather(metric_count).sum().item()

            for metric_name, metric_sum in self._train_metric_sums.items():
                total_metric_sum = self._nested_gather(metric_sum).sum().item()
                logs[metric_name] = round(total_metric_sum / max(total_metric_count, 1.0), 6)

            self._train_metric_sums = {}
            self._train_metric_updates = 0
            return logs

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss, outputs = super().compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
            if model.training:
                self._accumulate_train_metrics(outputs)
            return (loss, outputs) if return_outputs else loss

        def log(self, logs, start_time=None):
            if "loss" in logs:
                logs = {**logs, **self._consume_train_metrics()}
            super().log(logs, start_time=start_time)

        def _save(self, output_dir=None, state_dict=None):
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            print(f"Saving model checkpoint to {output_dir}")

            model_to_save = self.deepspeed.model if self.is_deepspeed_enabled else self.model.model
            model_to_save.save_pretrained(
                output_dir,
                safe_serialization=self.args.save_safetensors,
                state_dict={
                    key.removeprefix("model."): value
                    for key, value in state_dict.items()
                    if key.startswith("model.")
                },
            )

            if self.tokenizer is not None and self.is_world_process_zero():
                self.tokenizer.save_pretrained(
                    output_dir,
                    safe_serialization=self.args.save_safetensors,
                )

    train_dataset = RankingDataset(
        data_args=data_args,
        batch_size=training_args.per_device_train_batch_size,
    )
    data_collator = RankingDataCollator(
        tokenizer=tokenizer,
        query_max_length=data_args.q_max_len,
        doc_max_length=data_args.d_max_len,
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")) and not training_args.overwrite_output_dir:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    save_model_for_trainer(trainer=trainer, output_dir=training_args.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)
        torch.save(model_args, os.path.join(training_args.output_dir, "model_args.bin"))
        torch.save(training_args, os.path.join(training_args.output_dir, "training_args.bin"))
        torch.save(rl_args, os.path.join(training_args.output_dir, "rl_args.bin"))
    print("Training done.")


if __name__ == "__main__":
    main()
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    print("Success.")
