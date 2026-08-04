"""GRPO training entrypoint, plus the setup helpers shared with ``train_baseline.py``.

The two entrypoints differ only in which dataclasses they parse, which wrapper module
they build, and which trainer they hand it to. Everything around that -- launcher-flag
splitting, config resolution, output-dir guarding, logging, backbone/LoRA/tokenizer
loading, dataset+collator construction, gradient checkpointing, and the final save --
is identical and lives at module level here.
"""

import logging
import os
import pathlib
import sys

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoConfig, AutoModel, AutoTokenizer, HfArgumentParser, set_seed
from transformers import TrainingArguments as HFTrainingArguments
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
from utils import (
    BASE_CONFIG_SLOTS,
    parse_config_file,
    parse_config_from_base_overrides,
    resolve_gradient_checkpointing_kwargs,
    resolve_run_name_and_output_dir,
    save_model_for_trainer,
)


logger = logging.getLogger(__name__)

CONFIG_FILE_SUFFIXES = {".json", ".yaml", ".yml"}


def split_launcher_args(
    cli_args: list[str],
    base_slots: tuple[str, ...] = BASE_CONFIG_SLOTS,
) -> tuple[dict[str, str], list[str]]:
    """Peel the ``--base-<slot> <path>`` launcher flags off the CLI argv."""
    base_flag_to_slot = {f"--base-{slot}": slot for slot in base_slots}

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


def parse_arguments(
    parser: HfArgumentParser,
    base_slots: tuple[str, ...] = BASE_CONFIG_SLOTS,
    argv: list[str] | None = None,
) -> tuple:
    """Resolve dataclasses from a top-level YAML, from ``--base-<slot>`` flags, or from plain CLI.

    Also fills in ``run_name``/``output_dir`` from whichever config source was used.
    """
    base_overrides, cli_args = split_launcher_args(
        sys.argv[1:] if argv is None else argv,
        base_slots=base_slots,
    )

    config_path = None
    if cli_args and pathlib.Path(cli_args[0]).suffix.lower() in CONFIG_FILE_SUFFIXES:
        config_path = cli_args[0]
        parsed = parse_config_file(
            parser=parser,
            config_path=config_path,
            cli_args=cli_args[1:],
            base_overrides=base_overrides,
        )
    elif base_overrides:
        parsed = parse_config_from_base_overrides(
            parser=parser,
            base_overrides=base_overrides,
            cli_args=cli_args,
            base_slots=base_slots,
        )
    else:
        parsed = parser.parse_args_into_dataclasses(cli_args)

    training_args = next(args for args in parsed if isinstance(args, HFTrainingArguments))
    resolve_run_name_and_output_dir(
        config_path=config_path,
        base_overrides=base_overrides,
        training_args=training_args,
        base_slots=base_slots,
    )
    return parsed


def guard_output_dir(training_args: HFTrainingArguments) -> None:
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


def setup_logging(training_args: HFTrainingArguments, extra_parameters: dict | None = None) -> None:
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
    for label, parameters in (extra_parameters or {}).items():
        logger.info("%s parameters %s", label, parameters)


def load_backbone_and_tokenizer(model_args: ModelArguments, lora_args: LoraArguments):
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

    return backbone, tokenizer


def apply_gradient_checkpointing(model, training_args: HFTrainingArguments, lora_args: LoraArguments) -> None:
    if not training_args.gradient_checkpointing:
        return
    training_args.gradient_checkpointing_kwargs = resolve_gradient_checkpointing_kwargs(
        training_args=training_args,
        lora_args=lora_args,
    )
    if training_args.gradient_checkpointing_kwargs.get("use_reentrant", True):
        model.enable_input_require_grads()
    logger.info("Gradient checkpointing kwargs %s", training_args.gradient_checkpointing_kwargs)


def build_ranking_data(data_args: DataArguments, training_args: HFTrainingArguments, tokenizer):
    """Train dataset, optional held-out dev dataset, and the shared collator."""
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
    return train_dataset, eval_dataset, data_collator


def save_run_artifacts(trainer, training_args: HFTrainingArguments, tokenizer, **argument_objects) -> None:
    """Persist the model plus the pickled argument dataclasses under ``output_dir``."""
    save_model_for_trainer(trainer=trainer, output_dir=training_args.output_dir)
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(training_args.output_dir)
        torch.save(training_args, os.path.join(training_args.output_dir, "training_args.bin"))
        for name, argument_object in argument_objects.items():
            torch.save(argument_object, os.path.join(training_args.output_dir, f"{name}.bin"))
    print("Training done.")


def shutdown_distributed() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    print("Success.")


def main() -> None:
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments, RLArguments, MTEBEvalArguments)
    )
    model_args, data_args, training_args, lora_args, rl_args, mteb_eval_args = parse_arguments(
        parser=parser,
        base_slots=BASE_CONFIG_SLOTS,
    )

    guard_output_dir(training_args)
    setup_logging(
        training_args,
        {"Model": model_args, "RL": rl_args, "MTEB eval": mteb_eval_args},
    )

    set_seed(training_args.seed)

    backbone, tokenizer = load_backbone_and_tokenizer(model_args, lora_args)
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

    apply_gradient_checkpointing(model, training_args, lora_args)

    train_dataset, eval_dataset, data_collator = build_ranking_data(data_args, training_args, tokenizer)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=None,
        data_collator=data_collator,
    )
    mteb_callback = MTEBEvalCallback(mteb_eval_args)
    if mteb_callback.enabled:
        trainer.add_callback(mteb_callback.bind_trainer(trainer))

    trainer.train(resume_from_checkpoint=True if resume_checkpoint else None)

    save_run_artifacts(trainer, training_args, tokenizer, model_args=model_args, rl_args=rl_args)


if __name__ == "__main__":
    main()
    shutdown_distributed()
