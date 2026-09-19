#!/usr/bin/env python3
"""Collect G3-R2 QA evaluations into one comparison table.

Reads each run's ``qa_eval/summary.json`` under the G3-R2 output root and prints
the headline retrieval and generation metrics against the untrained E0
reference, which is the bar that matters: every round-1 G3 arm finished below
it.

    python scripts/analyze_g3_r2_results.py
    python scripts/analyze_g3_r2_results.py --format markdown >> docs/g3_rag_plan.md

Also reports the in-training tuning probe from each run's trainer_state.json so
a run that is still training can be compared on held-out answer MRR before its
final evaluation exists.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "checkpoints/iclr2027-g3-r2"
# The untrained reference. Read per scope from its own summary rather than
# hardcoded: E0's macro (0.3976 mrr@10), training-domain (0.4734) and held-out
# (0.3673) averages differ substantially, and comparing a training-domain run
# against the macro number silently inflates every delta by ~0.076.
BASELINE_NAME = "G3-E0 (untrained)"
BASELINE_SUMMARY = ROOT / "data/rag/eval/g3_e0/summary.json"
COLUMNS = (
    "answer_recall_at_5",
    "answer_recall_at_20",
    "answer_mrr_at_10",
    "generator_em",
    "generator_token_f1",
)


def load_baseline(scope: str) -> dict:
    if not BASELINE_SUMMARY.is_file():
        raise SystemExit(f"Missing untrained reference: {BASELINE_SUMMARY}")
    block = json.loads(BASELINE_SUMMARY.read_text()).get(scope) or {}
    missing = [key for key in COLUMNS if block.get(key) is None]
    if missing:
        raise SystemExit(f"Reference {BASELINE_SUMMARY} lacks {missing} under {scope}")
    return {key: block[key] for key in COLUMNS}


def load_summary(run_dir: Path, scope: str) -> dict | None:
    path = run_dir / "qa_eval" / "summary.json"
    if not path.is_file():
        return None
    try:
        summary = json.loads(path.read_text())
    except (ValueError, OSError) as exc:
        print(f"warning: unreadable {path}: {exc}", file=sys.stderr)
        return None
    block = summary.get(scope) or {}
    return {key: block.get(key) for key in COLUMNS}


def load_tuning_probe(run_dir: Path) -> tuple[int | None, float | None]:
    """Last logged held-out answer MRR, for runs still in flight."""
    states = sorted(run_dir.glob("checkpoint-*/trainer_state.json"))
    latest = run_dir / "trainer_state.json"
    if latest.is_file():
        states.append(latest)
    step = value = None
    for path in states:
        try:
            history = json.loads(path.read_text()).get("log_history", [])
        except (ValueError, OSError):
            continue
        for entry in history:
            if "tuning/answer_mrr_at_10" in entry:
                if step is None or entry.get("step", 0) >= step:
                    step = entry.get("step", 0)
                    value = entry["tuning/answer_mrr_at_10"]
    return step, value


def _cell(value, reference, width=9):
    if value is None:
        return " " * (width - 1) + "-"
    text = f"{value:.4f}"
    if reference is not None:
        text += f" ({value - reference:+.4f})"
    return text.rjust(width)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--scope",
        default="macro_average",
        choices=("macro_average", "training_domain_average", "held_out_average"),
    )
    parser.add_argument("--format", default="table", choices=("table", "markdown"))
    args = parser.parse_args()

    if not args.output_root.is_dir():
        print(f"No G3-R2 output root yet: {args.output_root}", file=sys.stderr)
        return 1

    runs = sorted(d for d in args.output_root.iterdir() if d.is_dir() and not d.name.startswith("."))
    if not runs:
        print(f"No runs under {args.output_root}", file=sys.stderr)
        return 1

    baseline = load_baseline(args.scope)
    rows = [(BASELINE_NAME, baseline, None, None)]
    for run_dir in runs:
        metrics = load_summary(run_dir, args.scope) or {key: None for key in COLUMNS}
        step, probe = load_tuning_probe(run_dir)
        rows.append((run_dir.name, metrics, step, probe))

    header = ["run", *COLUMNS, "tuning_mrr@10"]
    if args.format == "markdown":
        print(f"\n### QA suite, {args.scope} (delta vs untrained E0)\n")
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join(["---"] * len(header)) + "|")
        for name, metrics, step, probe in rows:
            cells = [
                "-" if metrics[key] is None
                else f"{metrics[key]:.4f}" + (
                    "" if name == BASELINE_NAME
                    else f" ({metrics[key] - baseline[key]:+.4f})"
                )
                for key in COLUMNS
            ]
            tuning = "-" if probe is None else f"{probe:.4f} @{step}"
            print(f"| {name} | " + " | ".join(cells) + f" | {tuning} |")
        return 0

    width = max(len(name) for name, *_ in rows) + 2
    print(f"\nQA suite, {args.scope} (delta vs untrained E0)\n")
    print("run".ljust(width) + "".join(column[:18].rjust(20) for column in COLUMNS) + "   tuning_mrr@10")
    print("-" * (width + 20 * len(COLUMNS) + 16))
    for name, metrics, step, probe in rows:
        reference = None if name == BASELINE_NAME else baseline
        cells = "".join(
            _cell(metrics[key], None if reference is None else reference[key], 20)
            for key in COLUMNS
        )
        tuning = "" if probe is None else f"   {probe:.4f} @{step}"
        print(name.ljust(width) + cells + tuning)
    print()
    missing = [name for name, metrics, _, _ in rows if all(v is None for v in metrics.values())]
    if missing:
        print(f"No final evaluation yet: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
