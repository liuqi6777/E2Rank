import argparse
import json
import logging
import os
import pathlib
import re
from copy import deepcopy

import torch
from transformers import HfArgumentParser, Trainer as HFTrainer

from config import DataArguments, LoraArguments, ModelArguments, RLArguments, TrainingArguments


logger = logging.getLogger(__name__)

BASE_CONFIG_SLOTS = ("train", "model", "grpo", "reward")


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


def resolve_config_inheritance_from_dict(
    config: dict,
    config_path: str,
    visited_paths: tuple[str, ...] = (),
) -> dict:
    config_path = os.path.abspath(config_path)
    if config_path in visited_paths:
        cycle = " -> ".join((*visited_paths, config_path))
        raise ValueError(f"Detected config inheritance cycle: {cycle}")

    local_config = deepcopy(config)
    base_entries = local_config.pop("_base_", [])
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
        base_config = load_raw_config_file(base_path)
        resolved_base_config = resolve_config_inheritance_from_dict(
            config=base_config,
            config_path=base_path,
            visited_paths=(*visited_paths, config_path),
        )
        merged_config = merge_config_dicts(merged_config, resolved_base_config)

    return merge_config_dicts(merged_config, local_config)


def apply_base_overrides(config_path: str, config: dict, base_overrides: dict[str, str]) -> dict:
    if not base_overrides:
        return config

    updated_config = deepcopy(config)
    base_entries = updated_config.get("_base_", [])
    if isinstance(base_entries, str):
        base_entries = [base_entries]
    elif base_entries is None:
        base_entries = []
    elif not isinstance(base_entries, list):
        raise ValueError(f"`_base_` must be a string or a list of strings: {config_path}")

    replaced_slots: set[str] = set()
    resolved_entries: list[str] = []
    for base_entry in base_entries:
        if not isinstance(base_entry, str):
            raise ValueError(f"Each `_base_` entry must be a string: {config_path}")

        base_slot = pathlib.Path(base_entry).parent.name
        if base_slot in base_overrides:
            resolved_entries.append(os.path.abspath(base_overrides[base_slot]))
            replaced_slots.add(base_slot)
        else:
            resolved_entries.append(base_entry)

    missing_slots = sorted(set(base_overrides) - replaced_slots)
    if missing_slots:
        raise ValueError(
            f"Unknown `_base_` override slot(s) for {config_path}: {', '.join(missing_slots)}"
        )

    updated_config["_base_"] = resolved_entries
    return updated_config


def resolve_slot_config_paths(base_overrides: dict[str, str]) -> list[str]:
    invalid_slots = sorted(set(base_overrides) - set(BASE_CONFIG_SLOTS))
    if invalid_slots:
        raise ValueError(f"Unsupported base config slot(s): {', '.join(invalid_slots)}")

    resolved_paths: list[str] = []
    for slot in BASE_CONFIG_SLOTS:
        path = base_overrides.get(slot)
        if path:
            resolved_paths.append(os.path.abspath(path))
    return resolved_paths


def resolve_config_from_base_overrides(base_overrides: dict[str, str]) -> dict:
    merged_config: dict = {}
    for config_path in resolve_slot_config_paths(base_overrides):
        resolved_config = resolve_config_inheritance(config_path)
        merged_config = merge_config_dicts(merged_config, resolved_config)
    return merged_config


def slugify_name(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    slug = re.sub(r"-{2,}", "-", slug)
    return slug.strip("-") or "run"


def build_auto_run_name(config_path: str | None, base_overrides: dict[str, str]) -> str:
    if base_overrides:
        parts: list[str] = []
        for slot in BASE_CONFIG_SLOTS:
            slot_path = base_overrides.get(slot)
            if slot_path:
                stem = pathlib.Path(slot_path).stem
                if stem == "default" and slot in {"train", "grpo"}:
                    continue
                parts.append(stem)
        if parts:
            return "__".join(slugify_name(part) for part in parts)

    if config_path:
        return slugify_name(pathlib.Path(config_path).stem)

    return "run"


def resolve_run_name_and_output_dir(
    config_path: str | None,
    base_overrides: dict[str, str],
    training_args: TrainingArguments,
) -> None:
    output_dir = (training_args.output_dir or "").strip()
    run_name = (training_args.run_name or "").strip() if training_args.run_name else ""

    output_dir_is_default = output_dir in {"", "trainer_output"}
    run_name_is_default = run_name in {"", "trainer_output"}

    if run_name_is_default and not output_dir_is_default:
        run_name = pathlib.Path(output_dir).name

    if not run_name_is_default and output_dir_is_default:
        output_dir = str(pathlib.Path("checkpoints") / run_name)

    if run_name_is_default and output_dir_is_default:
        run_name = build_auto_run_name(
            config_path=config_path,
            base_overrides=base_overrides,
        )
        output_dir = str(pathlib.Path("checkpoints") / run_name)

    training_args.run_name = run_name
    training_args.output_dir = output_dir


def parse_config_file(
    parser: HfArgumentParser,
    config_path: str,
    cli_args: list[str] | None = None,
    base_overrides: dict[str, str] | None = None,
) -> tuple[ModelArguments, DataArguments, TrainingArguments, LoraArguments, RLArguments]:
    root_config = load_raw_config_file(config_path)
    root_config = apply_base_overrides(
        config_path=config_path,
        config=root_config,
        base_overrides=base_overrides or {},
    )
    config = resolve_config_inheritance_from_dict(
        config=root_config,
        config_path=config_path,
    )
    if cli_args:
        override_config = parse_cli_overrides(
            dataclass_types=parser.dataclass_types,
            cli_args=cli_args,
        )
        config = merge_config_dicts(config, override_config)
    return parser.parse_dict(config)


def parse_config_from_base_overrides(
    parser: HfArgumentParser,
    base_overrides: dict[str, str],
    cli_args: list[str] | None = None,
) -> tuple[ModelArguments, DataArguments, TrainingArguments, LoraArguments, RLArguments]:
    config = resolve_config_from_base_overrides(base_overrides)
    if cli_args:
        override_config = parse_cli_overrides(
            dataclass_types=parser.dataclass_types,
            cli_args=cli_args,
        )
        config = merge_config_dicts(config, override_config)
    return parser.parse_dict(config)


def parse_cli_overrides(dataclass_types: list[type], cli_args: list[str]) -> dict:
    override_parser = HfArgumentParser(dataclass_types)
    for action in override_parser._actions:
        action.required = False
        if action.dest != "help":
            action.default = argparse.SUPPRESS

    namespace, remaining_args = override_parser.parse_known_args(cli_args)
    if remaining_args:
        joined_args = " ".join(remaining_args)
        raise ValueError(f"Unrecognized arguments: {joined_args}")

    return vars(namespace)


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
