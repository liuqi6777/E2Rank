#!/usr/bin/env python3

import argparse
import csv
import itertools
import os
import pathlib
import re
import shlex
import subprocess
import sys
from collections import OrderedDict


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import (
    apply_base_overrides,
    load_raw_config_file,
    resolve_config_from_base_overrides,
    resolve_config_inheritance_from_dict,
)


BASE_SLOT_TO_FLAG = {
    "train": "--base-train",
    "dataset": "--base-dataset",
    "model": "--base-model",
    "grpo": "--base-grpo",
    "reward": "--base-reward",
    "eval": "--base-eval",
}


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Expand a simple parameter grid and launch runs sequentially.",
    )
    parser.add_argument("config_path", nargs="?", help="Optional experiment config template.")
    parser.add_argument(
        "--set",
        dest="grid_specs",
        action="append",
        default=[],
        metavar="KEY=V1,V2",
        help="Grid spec. Values are comma-separated scalars.",
    )
    parser.add_argument(
        "--set-base",
        dest="base_grid_specs",
        action="append",
        default=[],
        metavar="SLOT=PATH1,PATH2",
        help="Grid over config slots. Supported slots: train, dataset, model, grpo, reward, eval.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional root directory for generated output_dir values.",
    )
    parser.add_argument(
        "--run-name-prefix",
        default=None,
        help="Optional prefix used for generated run_name values.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the expanded commands without launching them.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on the number of expanded runs.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Starting index used in generated run ids.",
    )
    parser.add_argument(
        "--launcher",
        default=str(REPO_ROOT / "scripts" / "run.sh"),
        help="Launcher script used for each expanded run.",
    )
    args, passthrough_args = parser.parse_known_args()
    if args.config_path and args.config_path.startswith("--"):
        passthrough_args = [args.config_path, *passthrough_args]
        args.config_path = None
    return args, passthrough_args


def parse_scalar(raw_value: str):
    text = raw_value.strip()
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None

    try:
        return int(text)
    except ValueError:
        pass

    try:
        return float(text)
    except ValueError:
        return text


def parse_multi_value(raw_values: str) -> list[str]:
    return [value.strip() for value in next(csv.reader([raw_values])) if value.strip()]


def parse_grid_specs(grid_specs: list[str]) -> OrderedDict[str, list]:
    parsed_specs: OrderedDict[str, list] = OrderedDict()
    for spec in grid_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid grid spec `{spec}`. Expected KEY=V1,V2.")
        key, raw_values = spec.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid grid spec `{spec}`. Key cannot be empty.")
        if "." in key:
            raise ValueError(f"Nested keys are not supported in grid specs: `{key}`")
        if key in parsed_specs:
            raise ValueError(f"Duplicate grid key: `{key}`")

        values = parse_multi_value(raw_values)
        if not values:
            raise ValueError(f"Invalid grid spec `{spec}`. At least one value is required.")

        parsed_specs[key] = [parse_scalar(value) for value in values]
    return parsed_specs


def parse_base_grid_specs(base_grid_specs: list[str]) -> OrderedDict[str, list[str]]:
    parsed_specs: OrderedDict[str, list[str]] = OrderedDict()
    for spec in base_grid_specs:
        if "=" not in spec:
            raise ValueError(f"Invalid base grid spec `{spec}`. Expected SLOT=PATH1,PATH2.")
        slot, raw_values = spec.split("=", 1)
        slot = slot.strip()
        if slot not in BASE_SLOT_TO_FLAG:
            supported = ", ".join(sorted(BASE_SLOT_TO_FLAG))
            raise ValueError(f"Unsupported `_base_` slot `{slot}`. Expected one of: {supported}")
        if slot in parsed_specs:
            raise ValueError(f"Duplicate `_base_` slot: `{slot}`")

        values = [str((REPO_ROOT / value).resolve()) if not os.path.isabs(value) else value for value in parse_multi_value(raw_values)]
        if not values:
            raise ValueError(f"Invalid base grid spec `{spec}`. At least one path is required.")
        parsed_specs[slot] = values
    return parsed_specs


def stringify_override(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    return str(value)


def slugify(value) -> str:
    text = stringify_override(value)
    text = text.replace(os.sep, "-")
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text)
    return text.strip("-") or "value"


def base_slug(path: str) -> str:
    return slugify(pathlib.Path(path).stem)


def build_run_suffix(
    index: int,
    base_overrides: OrderedDict[str, str],
    overrides: OrderedDict[str, object],
) -> str:
    parts = [f"run{index:03d}"]
    for slot, path in base_overrides.items():
        parts.append(f"{slot}-{base_slug(path)}")
    for key, value in overrides.items():
        parts.append(f"{key}-{slugify(value)}")
    return "__".join(parts)


def build_output_dir(base_output_dir: str, output_root: str | None, suffix: str) -> str:
    if output_root:
        return str(pathlib.Path(output_root) / suffix)
    return f"{base_output_dir}__{suffix}"


def resolve_base_config(config_path: str | None, base_overrides: dict[str, str]) -> dict:
    if config_path is None:
        return resolve_config_from_base_overrides(base_overrides)

    raw_config = load_raw_config_file(config_path)
    raw_config = apply_base_overrides(
        config_path=config_path,
        config=raw_config,
        base_overrides=base_overrides,
    )
    return resolve_config_inheritance_from_dict(
        config=raw_config,
        config_path=config_path,
    )


def main() -> int:
    args, passthrough_args = parse_args()
    if passthrough_args and passthrough_args[0] == "--":
        passthrough_args = passthrough_args[1:]

    launcher_path = pathlib.Path(args.launcher)
    if not launcher_path.is_absolute():
        launcher_path = (REPO_ROOT / launcher_path).resolve()
    if not launcher_path.is_file():
        raise FileNotFoundError(f"Launcher script not found: {launcher_path}")

    config_path: pathlib.Path | None = None
    if args.config_path:
        config_path = pathlib.Path(args.config_path)
        if not config_path.is_absolute():
            config_path = (REPO_ROOT / config_path).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")

    grid_specs = parse_grid_specs(args.grid_specs)
    base_grid_specs = parse_base_grid_specs(args.base_grid_specs)

    grid_keys = list(grid_specs.keys())
    base_grid_keys = list(base_grid_specs.keys())
    grid_values = list(grid_specs.values())
    base_grid_values = list(base_grid_specs.values())

    param_combinations = itertools.product(*grid_values) if grid_values else [()]
    base_combinations = itertools.product(*base_grid_values) if base_grid_values else [()]

    commands: list[tuple[list[str], OrderedDict[str, str], OrderedDict[str, object]]] = []
    run_offset = 0
    for base_combination in base_combinations:
        base_overrides = OrderedDict(zip(base_grid_keys, base_combination))
        resolved_config = resolve_base_config(str(config_path) if config_path else None, dict(base_overrides))
        config_label = config_path.stem if config_path else "run"
        base_run_name = args.run_name_prefix or resolved_config.get("run_name") or config_label
        base_output_dir = resolved_config.get("output_dir") or f"checkpoints/{base_run_name}"

        for combination in param_combinations:
            run_index = args.start_index + run_offset
            overrides = OrderedDict(zip(grid_keys, combination))
            suffix = build_run_suffix(run_index, base_overrides, overrides)
            run_name = f"{base_run_name}__{suffix}"
            output_dir = build_output_dir(str(base_output_dir), args.output_root, suffix)

            command = [
                "bash",
                str(launcher_path),
                "--run_name",
                run_name,
                "--output_dir",
                output_dir,
            ]
            if config_path:
                command.insert(2, str(config_path))
            for slot, path in base_overrides.items():
                command.extend([BASE_SLOT_TO_FLAG[slot], path])
            for key, value in overrides.items():
                command.extend([f"--{key}", stringify_override(value)])
            command.extend(passthrough_args)
            commands.append((command, base_overrides, overrides))
            run_offset += 1

    if args.max_runs is not None:
        commands = commands[: args.max_runs]

    if not commands:
        raise ValueError("No runs were generated.")

    if config_path:
        print(f"Expanded {len(commands)} run(s) from {config_path}.")
    else:
        print(f"Expanded {len(commands)} run(s) from base config slots.")
    for index, (command, base_overrides, overrides) in enumerate(commands, start=1):
        parts: list[str] = []
        parts.extend(f"{slot}={pathlib.Path(path).name}" for slot, path in base_overrides.items())
        parts.extend(f"{key}={value}" for key, value in overrides.items())
        summary = ", ".join(parts) if parts else "no grid overrides"
        print(f"[{index:02d}] {summary}")
        print(f"     {shlex.join(command)}")

    if args.dry_run:
        return 0

    for index, (command, _, _) in enumerate(commands, start=1):
        print(f"Launching run {index}/{len(commands)}")
        subprocess.run(command, check=True, cwd=REPO_ROOT)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
