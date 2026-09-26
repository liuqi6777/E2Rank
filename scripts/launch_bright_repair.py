#!/usr/bin/env python3
"""Distribute corrected BRIGHT evaluations across independent eight-GPU workers."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys

import eval_iclr2027_gpt_reasoning_06b as small
import eval_iclr2027_gpt_reasoning_4b as large
import eval_iclr2027_gpt_reasoning_bge_m3 as bge
from experiments import bright_gpt_reasoning as reasoning
from experiments import iclr2027
from run_g1_stability import SUBSETS

SPECS = {"4b": large.SPEC, "06b": small.SPEC, "bge-m3": bge.SPEC}


@dataclass
class Job:
    backbone: str
    row: dict
    cost: float

    @property
    def name(self):
        return f"{self.backbone}/{self.row['run_id']}"


def make_jobs(args):
    jobs = []
    for backbone, spec in SPECS.items():
        settings = getattr(args, "config_" + backbone.replace("-", "_")) or spec.settings
        names = reasoning.selected_runs(spec, list(reasoning.METHODS), list(reasoning.SEEDS))
        rows = reasoning.resolve_rows(spec, settings, names)
        cost = args.cost_4b if backbone == "4b" else 1.0
        jobs.extend(Job(backbone, row, cost) for row in rows)
    return jobs


def allocate(jobs, workers):
    """Longest estimated work first; deterministic across independent nodes."""
    slots = [[] for _ in range(workers)]
    loads = [0.0] * workers
    for job in sorted(jobs, key=lambda job: -job.cost):
        slot = min(range(workers), key=lambda i: (loads[i], i))
        slots[slot].append(job)
        loads[slot] += job.cost
    return slots, loads


def result_root(job):
    base = Path(job.row["config"]["output_dir"]) / "mteb_eval"
    return base / reasoning.RESULT_NAME / f"query-{reasoning.QUERY_SET}"


def read_result(root):
    paths = list(root.rglob("BrightRetrieval.json")) if root.exists() else []
    if len(paths) > 1:
        raise ValueError(f"Ambiguous BRIGHT results: {paths}")
    if not paths:
        return None, None, {}
    payload = json.loads(paths[0].read_text())
    scores = {}
    for entry in payload.get("scores", {}).get("standard", []):
        subset = entry.get("hf_subset")
        if subset not in SUBSETS or subset in scores:
            raise ValueError(f"Invalid/duplicate subset {subset!r}: {paths[0]}")
        score = float(entry["main_score"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid BRIGHT score: {paths[0]}")
        scores[subset] = entry
    return paths[0], payload, scores


def command(job, subsets):
    return [*reasoning.eval_command(job.row),
            "--run_kwargs", json.dumps({"eval_subsets": list(subsets)})]


@contextmanager
def job_lock(job):
    folder = Path(job.row["config"]["output_dir"]) / "mteb_eval"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".bright-repair.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another worker is evaluating {job.name}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_job(job, log_dir):
    with job_lock(job):
        _, _, scores = read_result(result_root(job))
        missing = [s for s in SUBSETS if s not in scores]
        if not missing:
            print(f"SKIP {job.name}: 12/12 complete", flush=True)
            return
        cmd = command(job, missing)
        log = log_dir / f"{job.backbone}-{job.row['run_id']}-gpt4-reasoning.log"
        print(f"RUN {job.name}: {len(missing)} domains; log={log}", flush=True)
        with log.open("a") as handle:
            handle.write("\n" + datetime.now(timezone.utc).isoformat() + "\n" + shlex.join(cmd) + "\n")
            handle.flush()
            completed = subprocess.run(cmd, cwd=iclr2027.ROOT, stdout=handle, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"Exit {completed.returncode}; see {log}")
        _, _, scores = read_result(result_root(job))
        if set(scores) != set(SUBSETS):
            raise RuntimeError(f"Incomplete output for {job.name}; see {log}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "check", "run"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--slot", type=int, help="Zero-based worker number (0..15 by default)")
    parser.add_argument("--cost-4b", type=float, default=4.0,
                        help="Estimated relative runtime for scheduling, not a measured speed ratio")
    parser.add_argument("--log-dir", type=Path, default=iclr2027.ROOT / "logs/bright-repair")
    for backbone in SPECS:
        parser.add_argument("--config-" + backbone, type=Path)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.cost_4b <= 0 or not math.isfinite(args.cost_4b):
        parser.error("workers and cost-4b must be positive")
    if args.slot is not None and not 0 <= args.slot < args.workers:
        parser.error("slot must be in [0, workers)")
    if args.action == "run" and args.slot is None:
        parser.error("run needs --slot; start one process on each eight-GPU node")
    slots, loads = allocate(make_jobs(args), args.workers)
    selected = range(args.workers) if args.slot is None else [args.slot]
    if args.action == "plan":
        for slot in selected:
            print(f"SLOT {slot:02d}: {len(slots[slot])} checkpoints, estimated load={loads[slot]:g}")
            for job in slots[slot]:
                print(f"  {job.name}: gpt4-reasoning")
        return 0
    # Check every assigned checkpoint before consuming any GPU time.
    for slot in selected:
        for job in slots[slot]:
            reasoning.check_model(job.row)
    if args.action == "check":
        print(f"Checked {sum(len(slots[s]) for s in selected)} checkpoints; no evaluation launched.")
        return 0
    # Imported lazily so plan/check work on CPU without importing MTEB.
    import torch
    if torch.cuda.device_count() != 8:
        raise ValueError("Each worker must see exactly eight GPUs; configure CUDA_VISIBLE_DEVICES or the scheduler")
    log_dir = args.log_dir / f"slot-{args.slot:02d}"
    log_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for job in slots[args.slot]:
        try:
            run_job(job, log_dir)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            failures.append({"job": job.name, "error": str(exc)})
            print(f"FAILED {job.name}: {exc}", file=sys.stderr, flush=True)
    (log_dir / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    print(f"Slot {args.slot}: {len(slots[args.slot]) - len(failures)} complete, {len(failures)} failed", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
