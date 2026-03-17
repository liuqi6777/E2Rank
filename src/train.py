import logging
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
        attn_implementation="flash_attention_2",
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
        query_reward_ndcg_k=rl_args.query_reward_ndcg_k,
        listwise_reward_ndcg_k=rl_args.listwise_reward_ndcg_k,
        listwise_loss_weight=rl_args.listwise_loss_weight,
        advantage_norm=rl_args.advantage_norm,
        query_relevance_scheme=rl_args.query_relevance_scheme,
        listwise_relevance_scheme=rl_args.listwise_relevance_scheme,
    )
    model.train()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    class Trainer(HFTrainer):
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
