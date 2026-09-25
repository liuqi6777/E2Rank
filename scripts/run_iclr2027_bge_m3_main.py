#!/usr/bin/env python3
"""Run the BGE-M3 main-table extension: 9 trainings and two BRIGHT query sets."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

from experiments import iclr2027
from experiments import bright_gpt_reasoning as reasoning
from eval_iclr2027_gpt_reasoning_bge_m3 import SPEC
from run_g1_r2 import read_bright, validate_final_model, verify_data

SEEDS = reasoning.SEEDS
TRAIN_METHODS = ("infonce", "lambdaloss", "reler")


def selected_rows(settings: Path, methods: list[str], seeds: list[int]) -> list[dict]:
    names = reasoning.selected_runs(SPEC, methods, seeds)
    rows = reasoning.resolve_rows(SPEC, settings, names)
    expected = {
        "model_name_or_path": "BAAI/bge-m3",
        "model_revision": SPEC.revision,
        "pooling_method": "cls",
        "padding_side": "right",
        "append_token": "none",
        "query_prompt_template": "{query}",
        "document_prompt_template": "{document}",
        "lora_enabled": False,
        "max_steps": 113,
        "learning_rate": 5e-6,
        "save_steps": 25,
        "per_device_train_batch_size": 16,
        "gradient_accumulation_steps": 1,
        "max_grad_norm": 0,
        "save_only_model": True,
        "load_best_model_at_end": False,
    }
    for row in rows:
        cfg = row["config"]
        mismatch = {key: (cfg.get(key), value) for key, value in expected.items()
                    if cfg.get(key) != value}
        if mismatch:
            raise ValueError(f"{row['run_id']}: main-table BGE-M3 config differs: {mismatch}")
        if row["objective"] == "rl" and cfg["rollout_seed"] != cfg["seed"]:
            raise ValueError(f"{row['run_id']}: rollout seed differs from training seed")
    return rows


def original_command(row: dict) -> list[str]:
    return iclr2027._post_mteb_command(row, iclr2027.BRIGHT_BENCHMARK, "bright")


def original_complete(row: dict) -> bool:
    return read_bright(row["config"]["output_dir"])[0] is not None


def evaluate(row: dict) -> None:
    if row["kind"] == "train":
        validate_final_model(row)
    if original_complete(row):
        print(f"{row['run_id']}: original query complete; skipping", flush=True)
    else:
        print(f"{row['run_id']}: evaluating original query", flush=True)
        subprocess.run(original_command(row), cwd=iclr2027.ROOT, check=True)
        if not original_complete(row):
            raise ValueError(f"{row['run_id']}: incomplete original-query BRIGHT result")
    if reasoning.result_score(row) is not None:
        print(f"{row['run_id']}: GPT-reasoning query complete; skipping", flush=True)
    else:
        print(f"{row['run_id']}: evaluating GPT-reasoning query", flush=True)
        subprocess.run(reasoning.eval_command(row), cwd=iclr2027.ROOT, check=True)
        if reasoning.result_score(row) is None:
            raise ValueError(f"{row['run_id']}: incomplete GPT-reasoning BRIGHT result")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "check", "status", "train", "eval"))
    parser.add_argument("--config", type=Path, default=SPEC.settings)
    parser.add_argument("--methods", nargs="+", choices=reasoning.METHODS,
                        default=list(reasoning.METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    args = parser.parse_args(argv)
    if len(set(args.methods)) != len(args.methods) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Select distinct methods and seeds")
    try:
        rows = selected_rows(args.config, args.methods, args.seeds)
        if args.action == "plan":
            for row in rows:
                cfg = row["config"]
                print(f"{row['run_id']}: {row['objective'] or 'E0'}, seed={cfg['seed']}, "
                      f"output={cfg['output_dir']}")
                print("  original: " + shlex.join(original_command(row)))
                print("  reasoning: " + shlex.join(reasoning.eval_command(row)))
            return 0
        if args.action == "status":
            for row in rows:
                model = row["kind"] == "evaluation"
                if not model:
                    try:
                        validate_final_model(row)
                        model = True
                    except (ValueError, OSError, KeyError):
                        pass
                print(f"{row['run_id']}: model={'ready' if model else 'missing'}, "
                      f"original={'complete' if original_complete(row) else 'missing'}, "
                      f"reasoning={'complete' if reasoning.result_score(row) is not None else 'missing'}")
            return 0
        if args.action in {"check", "train"}:
            data_hash = verify_data(rows)
            print(f"Prepared ReasonRank SHA256: {data_hash}", flush=True)
        if args.action == "check":
            print(f"Checked {len(rows)} selected runs; no GPU job launched.")
            return 0
        if args.action == "train" and any(row["kind"] == "train" for row in rows):
            import torch
            if torch.cuda.device_count() != 8 or not torch.cuda.is_bf16_supported():
                raise ValueError("Training needs exactly eight visible BF16 CUDA GPUs")
        for row in rows:
            if args.action == "train" and row["kind"] == "train":
                out = Path(row["config"]["output_dir"])
                if out.exists():
                    validate_final_model(row)
                    print(f"{row['run_id']}: trained model exists; skipping training", flush=True)
                else:
                    print(f"{row['run_id']}: training", flush=True)
                    iclr2027.run_simple_baselines(SPEC.suite, args.config, [row["run_id"]], "train")
            evaluate(row)
        return 0
    except (ValueError, OSError, KeyError, TypeError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
