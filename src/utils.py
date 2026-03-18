import json
import logging
import os
import pathlib
from copy import deepcopy

import torch
from transformers import HfArgumentParser, Trainer as HFTrainer

from config import DataArguments, LoraArguments, ModelArguments, RLArguments, TrainingArguments


logger = logging.getLogger(__name__)


def load_raw_config_file(config_path: str) -> dict:
    config_path = os.path.abspath(config_path)
    suffix = pathlib.Path(config_path).suffix.lower()

    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "Reading YAML configs requires PyYAML. Install it with `pip install pyyaml`."
        ) from exc

    with open(config_path, "r", encoding="utf-8") as fp:
        if suffix == ".json":
            config = json.load(fp)
        elif suffix in {".yaml", ".yml"}:
            config = yaml.safe_load(fp) or {}
        else:
            raise ValueError(
                f"Unsupported config file format: {config_path}. "
                "Expected one of: .json, .yaml, .yml."
            )

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {config_path}")

    return config


def merge_config_dicts(base_config: dict, override_config: dict) -> dict:
    merged = deepcopy(base_config)

    for key, value in override_config.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = merge_config_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)

    return merged


def resolve_config_inheritance(config_path: str, visited_paths: tuple[str, ...] = ()) -> dict:
    config_path = os.path.abspath(config_path)
    if config_path in visited_paths:
        cycle = " -> ".join((*visited_paths, config_path))
        raise ValueError(f"Detected config inheritance cycle: {cycle}")

    config = load_raw_config_file(config_path)
    base_entries = config.pop("_base_", [])
    if isinstance(base_entries, str):
        base_entries = [base_entries]
    elif base_entries is None:
        base_entries = []
    elif not isinstance(base_entries, list):
        raise ValueError(f"`_base_` must be a string or a list of strings: {config_path}")

    merged_config: dict = {}
    config_dir = os.path.dirname(config_path)

    for base_entry in base_entries:
        if not isinstance(base_entry, str):
            raise ValueError(f"Each `_base_` entry must be a string: {config_path}")
        base_path = base_entry if os.path.isabs(base_entry) else os.path.join(config_dir, base_entry)
        base_config = resolve_config_inheritance(base_path, visited_paths=(*visited_paths, config_path))
        merged_config = merge_config_dicts(merged_config, base_config)

    return merge_config_dicts(merged_config, config)


def parse_config_file(
    parser: HfArgumentParser,
    config_path: str,
) -> tuple[ModelArguments, DataArguments, TrainingArguments, LoraArguments, RLArguments]:
    config = resolve_config_inheritance(config_path)
    return parser.parse_dict(config)


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
