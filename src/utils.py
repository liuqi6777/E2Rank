import argparse
import json
import logging
import os
import pathlib
import re
from copy import deepcopy

import torch
from transformers import HfArgumentParser, Trainer as HFTrainer

from config import LoraArguments, TrainingArguments


logger = logging.getLogger(__name__)

BASE_CONFIG_SLOTS = ("train", "dataset", "model", "grpo", "reward", "eval")
BASELINE_CONFIG_SLOTS = ("train", "dataset", "model", "baseline", "eval")
MODEL_DEFAULT_TRAIN_KEY = "_default_train_"


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


def normalize_base_entries(base_entries, config_path: str) -> list:
    """Coerce a `_base_` value (absent / string / list) into a list."""
    if base_entries is None:
        return []
    if isinstance(base_entries, str):
        return [base_entries]
    if isinstance(base_entries, list):
        return base_entries
    raise ValueError(f"`_base_` must be a string or a list of strings: {config_path}")


def resolve_config_inheritance(config_path: str, visited_paths: tuple[str, ...] = ()) -> dict:
    return resolve_config_inheritance_from_dict(
        config=load_raw_config_file(config_path),
        config_path=config_path,
        visited_paths=visited_paths,
    )


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
    base_entries = normalize_base_entries(local_config.pop("_base_", []), config_path)
    # This is a composition directive, not a dataclass argument. It is consumed by
    # the slot resolver when no explicit train config was supplied.
    local_config.pop(MODEL_DEFAULT_TRAIN_KEY, None)

    merged_config: dict = {}
    config_dir = os.path.dirname(config_path)

    for base_entry in base_entries:
        if not isinstance(base_entry, str):
            raise ValueError(f"Each `_base_` entry must be a string: {config_path}")
        base_path = base_entry if os.path.isabs(base_entry) else os.path.join(config_dir, base_entry)
        resolved_base_config = resolve_config_inheritance(
            base_path,
            visited_paths=(*visited_paths, config_path),
        )
        merged_config = merge_config_dicts(merged_config, resolved_base_config)

    return merge_config_dicts(merged_config, local_config)


def resolve_model_default_train_config(model_config_path: str) -> str | None:
    """Return the train preset declared by a model config, resolved from that file."""
    model_config_path = os.path.abspath(model_config_path)
    config = load_raw_config_file(model_config_path)
    default_train = config.get(MODEL_DEFAULT_TRAIN_KEY)
    if default_train is None:
        return None
    if not isinstance(default_train, str) or not default_train.strip():
        raise ValueError(
            f"`{MODEL_DEFAULT_TRAIN_KEY}` must be a non-empty string: {model_config_path}"
        )

    default_train_path = pathlib.Path(default_train)
    if not default_train_path.is_absolute():
        default_train_path = pathlib.Path(model_config_path).parent / default_train_path
    default_train_path = default_train_path.resolve()
    if not default_train_path.is_file():
        raise ValueError(
            f"Model default train config does not exist: {default_train_path} "
            f"(declared by {model_config_path})"
        )
    return str(default_train_path)


def apply_model_default_train_to_root(config_path: str, config: dict) -> dict:
    """Inject a model's default train base into a top-level `_base_` composition."""
    updated_config = deepcopy(config)
    base_entries = normalize_base_entries(updated_config.get("_base_", []), config_path)
    config_dir = pathlib.Path(config_path).resolve().parent

    slots: dict[str, tuple[int, str]] = {}
    for index, base_entry in enumerate(base_entries):
        if not isinstance(base_entry, str):
            raise ValueError(f"Each `_base_` entry must be a string: {config_path}")
        base_path = pathlib.Path(base_entry)
        resolved_path = base_path if base_path.is_absolute() else config_dir / base_path
        slots[resolved_path.parent.name] = (index, str(resolved_path))

    if "train" in slots or "model" not in slots:
        return updated_config

    model_index, model_path = slots["model"]
    default_train = resolve_model_default_train_config(model_path)
    if default_train:
        base_entries.insert(model_index, default_train)
        updated_config["_base_"] = base_entries
    return updated_config


def apply_base_overrides(config_path: str, config: dict, base_overrides: dict[str, str]) -> dict:
    if not base_overrides:
        return config

    updated_config = deepcopy(config)
    base_entries = normalize_base_entries(updated_config.get("_base_", []), config_path)

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


def resolve_slot_config_paths(
    base_overrides: dict[str, str],
    base_slots: tuple[str, ...] = BASE_CONFIG_SLOTS,
) -> list[str]:
    invalid_slots = sorted(set(base_overrides) - set(base_slots))
    if invalid_slots:
        raise ValueError(f"Unsupported base config slot(s): {', '.join(invalid_slots)}")

    resolved_paths: list[str] = []
    for slot in base_slots:
        path = base_overrides.get(slot)
        if path:
            resolved_paths.append(os.path.abspath(path))
    return resolved_paths


def resolve_config_from_base_overrides(
    base_overrides: dict[str, str],
    base_slots: tuple[str, ...] = BASE_CONFIG_SLOTS,
) -> dict:
    base_overrides = dict(base_overrides)
    if "train" not in base_overrides and "model" in base_overrides:
        default_train = resolve_model_default_train_config(base_overrides["model"])
        if default_train:
            base_overrides["train"] = default_train

    merged_config: dict = {}
    for config_path in resolve_slot_config_paths(base_overrides, base_slots=base_slots):
        resolved_config = resolve_config_inheritance(config_path)
        merged_config = merge_config_dicts(merged_config, resolved_config)
    return merged_config


def slugify_name(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    slug = re.sub(r"-{2,}", "-", slug)
    return slug.strip("-") or "run"


def build_auto_run_name(
    config_path: str | None,
    base_overrides: dict[str, str],
    base_slots: tuple[str, ...] = BASE_CONFIG_SLOTS,
) -> str:
    if base_overrides:
        parts: list[str] = []
        for slot in base_slots:
            slot_path = base_overrides.get(slot)
            if slot_path:
                stem = pathlib.Path(slot_path).stem
                if stem == "default":
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
    base_slots: tuple[str, ...] = BASE_CONFIG_SLOTS,
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
            base_slots=base_slots,
        )
        output_dir = str(pathlib.Path("checkpoints") / run_name)

    training_args.run_name = run_name
    training_args.output_dir = output_dir


def _parse_dict_with_cli_overrides(
    parser: HfArgumentParser,
    config: dict,
    cli_args: list[str] | None,
) -> tuple:
    if cli_args:
        override_config = parse_cli_overrides(
            dataclass_types=parser.dataclass_types,
            cli_args=cli_args,
        )
        config = merge_config_dicts(config, override_config)
    return parser.parse_dict(config)


def parse_config_file(
    parser: HfArgumentParser,
    config_path: str,
    cli_args: list[str] | None = None,
    base_overrides: dict[str, str] | None = None,
) -> tuple:
    root_config = apply_base_overrides(
        config_path=config_path,
        config=load_raw_config_file(config_path),
        base_overrides=base_overrides or {},
    )
    root_config = apply_model_default_train_to_root(config_path, root_config)
    config = resolve_config_inheritance_from_dict(
        config=root_config,
        config_path=config_path,
    )
    return _parse_dict_with_cli_overrides(parser, config, cli_args)


def parse_config_from_base_overrides(
    parser: HfArgumentParser,
    base_overrides: dict[str, str],
    cli_args: list[str] | None = None,
    base_slots: tuple[str, ...] = BASE_CONFIG_SLOTS,
) -> tuple:
    config = resolve_config_from_base_overrides(base_overrides, base_slots=base_slots)
    return _parse_dict_with_cli_overrides(parser, config, cli_args)


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


__all__ = [
    "BASE_CONFIG_SLOTS",
    "BASELINE_CONFIG_SLOTS",
    "MODEL_DEFAULT_TRAIN_KEY",
    "apply_base_overrides",
    "apply_model_default_train_to_root",
    "build_auto_run_name",
    "get_deepspeed_zero_stage",
    "load_raw_config_file",
    "merge_config_dicts",
    "normalize_base_entries",
    "parse_cli_overrides",
    "parse_config_file",
    "parse_config_from_base_overrides",
    "resolve_config_from_base_overrides",
    "resolve_config_inheritance",
    "resolve_config_inheritance_from_dict",
    "resolve_gradient_checkpointing_kwargs",
    "resolve_model_default_train_config",
    "resolve_run_name_and_output_dir",
    "resolve_slot_config_paths",
    "save_model_for_trainer",
]
