import logging
import os
import pathlib
import sys

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import HfArgumentParser, set_seed
from transformers.trainer_utils import get_last_checkpoint

from config import (
    DataArguments,
    LoraArguments,
    ModelArguments,
    MTEBEvalArguments,
    RLArguments,
    TrainingArguments,
)
from grpo import GRPOModel
from grpo_trainer import GRPOTrainer, restore_grpo_state
from mteb_eval_callback import MTEBEvalCallback
from ranking_data import RankingDataCollator, RankingDataset
from ranking_eval import ranking_compute_metrics
from utils import (
    parse_config_file,
    parse_config_from_base_overrides,
    resolve_run_name_and_output_dir,
    resolve_gradient_checkpointing_kwargs,
    save_model_for_trainer,
)


logger = logging.getLogger(__name__)


def split_launcher_args(cli_args: list[str]) -> tuple[dict[str, str], list[str]]:
    base_flag_to_slot = {
        "--base-train": "train",
        "--base-dataset": "dataset",
        "--base-model": "model",
        "--base-grpo": "grpo",
        "--base-reward": "reward",
        "--base-eval": "eval",
    }

    base_overrides: dict[str, str] = {}
    passthrough_args: list[str] = []
    index = 0
    while index < len(cli_args):
        arg = cli_args[index]
        if arg in base_flag_to_slot:
            if index + 1 >= len(cli_args):
                raise ValueError(f"Expected a YAML path after {arg}")
            base_overrides[base_flag_to_slot[arg]] = cli_args[index + 1]
            index += 2
            continue

        passthrough_args.append(arg)
        index += 1

    return base_overrides, passthrough_args


def main() -> None:
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments, RLArguments, MTEBEvalArguments)
    )

    base_overrides, cli_args = split_launcher_args(sys.argv[1:])

    if cli_args and pathlib.Path(cli_args[0]).suffix.lower() in {".json", ".yaml", ".yml"}:
        config_path = cli_args[0]
        model_args, data_args, training_args, lora_args, rl_args, mteb_eval_args = parse_config_file(
            parser=parser,
            config_path=config_path,
            cli_args=cli_args[1:],
            base_overrides=base_overrides,
        )
    elif base_overrides:
        config_path = None
        model_args, data_args, training_args, lora_args, rl_args, mteb_eval_args = parse_config_from_base_overrides(
            parser=parser,
            base_overrides=base_overrides,
            cli_args=cli_args,
        )
    else:
        config_path = None
        model_args, data_args, training_args, lora_args, rl_args, mteb_eval_args = (
            parser.parse_args_into_dataclasses(cli_args)
        )

    resolve_run_name_and_output_dir(
        config_path=config_path,
        base_overrides=base_overrides,
        training_args=training_args,
    )

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
    logger.info("MTEB eval parameters %s", mteb_eval_args)

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
        rl_args=rl_args,
    )
    model.train()

    # Resolved before the trainer is built so the exploration scale can be restored while
    # the parameter is still whole (DeepSpeed partitions it during trainer construction).
    resume_checkpoint = None
    if not training_args.overwrite_output_dir and os.path.isdir(training_args.output_dir):
        resume_checkpoint = get_last_checkpoint(training_args.output_dir)
    restore_grpo_state(model, resume_checkpoint)

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = resolve_gradient_checkpointing_kwargs(
            training_args=training_args,
            lora_args=lora_args,
        )
        if training_args.gradient_checkpointing_kwargs.get("use_reentrant", True):
            model.enable_input_require_grads()
        logger.info("Gradient checkpointing kwargs %s", training_args.gradient_checkpointing_kwargs)

    train_dataset = RankingDataset(
        data_args=data_args,
        batch_size=training_args.per_device_train_batch_size,
        split="train",
    )
    # Held-out dev split for model selection, so smoothing/LR are never tuned on MTEB.
    eval_dataset = None
    if data_args.dev_samples_per_source > 0:
        eval_dataset = RankingDataset(
            data_args=data_args,
            batch_size=training_args.per_device_eval_batch_size,
            split="dev",
        )
    data_collator = RankingDataCollator(
        tokenizer=tokenizer,
        query_max_length=data_args.q_max_len,
        doc_max_length=data_args.d_max_len,
        relevance_scheme=data_args.relevance_scheme,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=ranking_compute_metrics if eval_dataset is not None else None,
        data_collator=data_collator,
    )
    mteb_callback = MTEBEvalCallback(mteb_eval_args)
    if mteb_callback.enabled:
        trainer.add_callback(mteb_callback.bind_trainer(trainer))

    trainer.train(resume_from_checkpoint=True if resume_checkpoint else None)

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
