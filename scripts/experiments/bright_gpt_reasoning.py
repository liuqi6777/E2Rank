"""Shared launcher for the paper main table's BRIGHT GPT-reasoning queries."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from experiments import iclr2027
from run_g1_r2 import validate_final_model
from run_g1_stability import SUBSETS

SEEDS = (42, 3407, 2026)
METHODS = ("e0", "infonce", "lambdaloss", "reler")
RESULT_NAME = "bright_gpt_reasoning"
QUERY_SET = "gpt-reasoning"


@dataclass(frozen=True)
class EvalSpec:
    suite: Path
    settings: Path
    model: str
    revision: str | None
    lora: bool
    run_stems: dict[str, str]
    reler_contract: dict[str, object]


def selected_runs(spec: EvalSpec, methods: list[str], seeds: list[int]) -> list[str]:
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("Select distinct methods")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Select distinct seeds")
    names = []
    for method in methods:
        stem = spec.run_stems[method]
        if method == "e0":
            if 42 in seeds:
                names.append(stem)
        else:
            names.extend(stem + (f"-Seed{seed}" if seed != 42 else "") for seed in seeds)
    if not names:
        raise ValueError("No main-table runs selected")
    return names


def resolve_rows(spec: EvalSpec, settings: Path, names: list[str]) -> list[dict]:
    suite = iclr2027.apply_settings(iclr2027.load_suite(spec.suite), settings)
    rows = []
    for name in names:
        if name not in suite["runs"]:
            raise ValueError(f"Missing paper main-table run in suite: {name}")
        row = iclr2027.resolve_run(suite, spec.suite, name, nproc=8)
        config = row["config"]
        expected_kind = "evaluation" if name == spec.run_stems["e0"] else "train"
        if row["kind"] != expected_kind:
            raise ValueError(f"{name}: expected {expected_kind}, got {row['kind']}")
        if config["model_name_or_path"] != spec.model or config.get("model_revision") != spec.revision:
            raise ValueError(f"{name}: backbone or revision differs from the main table")
        if row["kind"] == "train" and config.get("lora_enabled") != spec.lora:
            raise ValueError(f"{name}: unexpected fine-tuning mode")
        if name.startswith(spec.run_stems["reler"]):
            mismatch = {
                key: (config.get(key), value)
                for key, value in spec.reler_contract.items()
                if config.get(key) != value
            }
            if mismatch:
                raise ValueError(f"{name}: main-table RELER recipe differs: {mismatch}")
        rows.append(row)
    return rows


def result_root(row: dict) -> Path:
    return Path(row["config"]["output_dir"]) / "mteb_eval" / RESULT_NAME / f"query-{QUERY_SET}"


def result_score(row: dict) -> float | None:
    paths = list(result_root(row).rglob("BrightRetrieval.json"))
    if not paths:
        return None
    if len(paths) != 1:
        raise ValueError(f"{row['run_id']}: ambiguous GPT-reasoning results: {paths}")
    payload = json.loads(paths[0].read_text())
    entries = payload.get("scores", {}).get("standard", [])
    scores = {}
    for entry in entries:
        subset = entry.get("hf_subset")
        if subset in scores:
            raise ValueError(f"{row['run_id']}: duplicate BRIGHT subset {subset}")
        if subset in SUBSETS:
            value = float(entry["main_score"])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{row['run_id']}: invalid score for {subset}: {value}")
            scores[subset] = value
    if set(scores) != set(SUBSETS):
        raise ValueError(f"{row['run_id']}: incomplete GPT-reasoning result ({len(scores)}/12 subsets): {paths[0]}")
    return statistics.mean(scores.values()) * 100


def check_model(row: dict) -> None:
    if row["kind"] == "evaluation":
        return  # The pinned, unadapted E0 is downloaded by the evaluation entrypoint.
    out = Path(row["config"]["output_dir"])
    if row["config"].get("lora_enabled"):
        if not (out / "adapter_config.json").is_file():
            raise ValueError(f"Missing LoRA adapter: {out}")
        merged = iclr2027._merged_lora_output_dir(row)
        merged_row = dict(row, config=dict(row["config"], output_dir=str(merged)))
        validate_final_model(merged_row)
    else:
        validate_final_model(row)


def eval_command(row: dict) -> list[str]:
    command = iclr2027._post_mteb_command(row, iclr2027.BRIGHT_BENCHMARK, RESULT_NAME)
    return [*command, "--bright_query_set", QUERY_SET]


def main(spec: EvalSpec, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "check", "eval"))
    parser.add_argument("--config", type=Path, default=spec.settings)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    args = parser.parse_args(argv)
    try:
        names = selected_runs(spec, args.methods, args.seeds)
        rows = resolve_rows(spec, args.config, names)
        jobs = []
        for row in rows:
            command = eval_command(row)
            if args.action == "plan":
                print(f"{row['run_id']}: {shlex.join(command)}")
                continue
            check_model(row)
            score = result_score(row)
            jobs.append((row, command, score))
        for row, command, score in jobs:
            if score is not None:
                print(f"{row['run_id']}: complete, Avg.={score:.2f}; skipping")
                continue
            print(f"{row['run_id']}: ready -> {result_root(row)}", flush=True)
            if args.action == "eval":
                subprocess.run(command, cwd=iclr2027.ROOT, check=True)
                score = result_score(row)
                if score is None:
                    raise ValueError(f"{row['run_id']}: evaluation produced no result")
                print(f"{row['run_id']}: complete, Avg.={score:.2f}", flush=True)
        return 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
