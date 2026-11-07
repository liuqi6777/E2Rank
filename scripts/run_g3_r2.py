#!/usr/bin/env python3
"""G3-R2 RAG runs: relevance-scheme factorial plus the graded-nDCG RL arm.

``run_simple_baselines`` cannot drive G3 -- it dereferences ``cfg['data_path']``
(scripts/experiments/iclr2027.py:560), a key no RAG config carries, and adding
one would trip the ``allow_extra_keys=False`` parse in src/utils.py. This driver
uses the full-preflight path instead: resolve -> blockers -> launch.

    python scripts/run_g3_r2.py check
    python scripts/run_g3_r2.py train --runs G3-R2-CL-AnswerMasked
    python scripts/run_g3_r2.py eval  --runs G3-R2-CL-AnswerMasked --retrieval-only
    # Rebuttal E1: four paper recipes, two new seeds (eight runs).
    python scripts/run_g3_r2.py check --seeds 3407 2026
    python scripts/run_g3_r2.py train --seeds 3407
    python scripts/run_g3_r2.py eval --seeds 3407 2026
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiments import iclr2027 as experiments  # noqa: E402


SUITE = experiments.ROOT / "configs/experiments/iclr2027/suite_g3_r2.yaml"
SETTINGS = experiments.ROOT / "configs/experiments_g3_r2.yaml"
LEGACY_RUNS = [
    "G3-R2-CL-Binary",
    "G3-R2-CL-AnswerMasked",
    "G3-R2-CL-Graded",
    "G3-R2-CL-Strong-AnswerMasked",
    "G3-R2-RL-GradedNDCG",
    "G3-R2-RL-BinaryNDCG",
    "G3-R2-RL-MRR-Graded",
    "G3-R2-ShortRL-CP-Graded",
    "G3-R2-ShortRL-SF-Graded",
    "G3-R2-RL-GradedNDCG-Anchor010",
    "G3-R2-RL-GradedNDCG-Anchor050",
    "G3-R2-RL-GradedNDCG-LRHalf",
    "G3-R2-RL-GradedNDCG-Seed3407",
    "G3-R2-RL-GradedNDCG-Seed2026",
    "G3-R2-RL-GradedNDCG-LRQuarter",
    "G3-R2-CL-AnswerMasked-Anchor050",
    "G3-R2-RL-AnswerF1",
    "G3-R2-RL-AnswerF1-Anchor050",
    "G3-R2-RL-AnswerEM-Anchor050",
    "G3-R2-RL-MixedNDCGF1-Anchor050",
    "G3-R2-RL-MixedNDCGEM-Anchor050",
]
SEEDS = (42, 3407, 2026)
PAPER_RUNS = [
    "G3-R2-CL-AnswerMasked-Anchor050",
    "G3-R2-RL-GradedNDCG-Anchor050",
    "G3-R2-RL-AnswerF1-Anchor050",
    "G3-R2-RL-MixedNDCGF1-Anchor050",
]
ALL_RUNS = LEGACY_RUNS + [
    f"{name}-Seed{seed}" for seed in SEEDS[1:] for name in PAPER_RUNS
]


def select_runs(runs=None, seeds=None):
    """Preserve legacy selection; --seeds defaults to the four E1 recipes."""
    if seeds is None:
        return list(dict.fromkeys(LEGACY_RUNS if runs is None else runs))
    selected = []
    for seed in dict.fromkeys(seeds):
        for name in PAPER_RUNS if runs is None else runs:
            # Accept either the base recipe or an explicit repeat as input.
            base = name.rsplit("-Seed", 1)[0] if "-Seed" in name else name
            candidate = base if seed == 42 else f"{base}-Seed{seed}"
            if candidate not in ALL_RUNS:
                raise ValueError(f"No registered seed {seed} for {base}")
            if candidate not in selected:
                selected.append(candidate)
    return selected


def _resolve(args, name):
    suite = experiments.apply_settings(experiments.load_suite(SUITE), args.config)
    return suite, experiments.resolve_run(suite, SUITE, name, nproc=args.gpus)


def _evaluate(resolved, args):
    output_dir = Path(resolved["config"]["output_dir"])
    command = [
        sys.executable,
        "src/eval_rag.py",
        "--checkpoint",
        str(output_dir),
        "--output-dir",
        str(output_dir / "qa_eval"),
        "--retrieval-k",
        str(args.retrieval_k),
    ]
    if args.retrieval_only:
        command.append("--retrieval-only")
    else:
        command += ["--generator-endpoint", args.generator_endpoint]
        # Recorded for cost accounting; must match the server's actual TP size.
        command += ["--generator-gpu-count", str(args.generator_tensor_parallel)]
    if args.overwrite:
        # A truncated evaluation leaves partial per-dataset shards behind.
        command.append("--overwrite")
    env = dict(os.environ, PYTHONPATH="src")
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=experiments.ROOT, env=env, check=True)


def _smoke(resolved, args):
    """Three optimizer steps on a throwaway output dir.

    A queue slot on this cluster is scarcer than the two minutes this costs, so
    every real launch validates the loss path, the index shards and the tuning
    probe before committing to a full epoch.
    """
    import json
    import tempfile

    config = dict(resolved["config"])
    config.update(
        {
            "rag_max_train_samples": 256,
            "rag_tuning_eval_samples": 64,
            "max_steps": 3,
            "num_train_epochs": 1,
            "save_strategy": "no",
            "save_steps": 3,
            "report_to": "none",
            "overwrite_output_dir": True,
            "output_dir": tempfile.mkdtemp(prefix="g3-smoke-"),
        }
    )
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(config, handle)
        config_path = handle.name
    command = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={args.gpus}",
        "src/train_rag.py",
        config_path,
    ]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=experiments.ROOT, env=dict(os.environ), check=True)
    print(f"smoke OK: {resolved['config']['run_name']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", nargs="?", default="check", choices=["check", "smoke", "train", "eval"]
    )
    parser.add_argument("--runs", nargs="+", choices=ALL_RUNS)
    parser.add_argument(
        "--seeds", "--seed", nargs="+", type=int, choices=SEEDS,
        help="Select paired training/data seeds; defaults to the four paper recipes when --runs is omitted",
    )
    parser.add_argument("--config", type=Path, default=SETTINGS)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--retrieval-k", type=int, default=20)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Re-score over a partial evaluation")
    parser.add_argument("--generator-endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--generator-tensor-parallel", type=int, default=4)
    args = parser.parse_args()
    runs = select_runs(args.runs, args.seeds)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.action in {"train", "smoke"}:
        import torch

        if torch.cuda.device_count() != args.gpus:
            raise ValueError(f"Expose exactly {args.gpus} CUDA GPUs (CUDA_VISIBLE_DEVICES)")

    blocked = 0
    for name in runs:
        suite, resolved = _resolve(args, name)
        config = resolved["config"]
        print(
            f"{name}: output={config['output_dir']} "
            f"seed={config['seed']} data_seed={config['data_seed']} "
            f"budget={resolved['protocol']['training_budget']} "
            f"epochs={config.get('num_train_epochs')} "
            f"micro={config['per_device_train_batch_size']} "
            f"accum={config['gradient_accumulation_steps']} "
            f"scheme={config.get('rag_relevance_scheme')} "
            f"split={config.get('rag_split')}",
            flush=True,
        )
        if args.action == "eval":
            _evaluate(resolved, args)
            continue
        if args.action == "smoke":
            _smoke(resolved, args)
            continue
        # blockers() walks the candidate -> index -> qrels -> corpus hash chain,
        # which re-hashes the 2.9 GB candidate pool.
        problems = experiments.blockers(suite, resolved)
        for issue in problems:
            print(f"  - {issue}", flush=True)
        blocked += bool(problems)
        if args.action == "train" and not problems:
            experiments.launch(suite, resolved, args.gpus)
    return 2 if blocked else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
