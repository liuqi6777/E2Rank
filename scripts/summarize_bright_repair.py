#!/usr/bin/env python3
"""Summarize the 30 corrected BRIGHT GPT-4-query evaluations without loading models."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from statistics import mean, stdev
import sys

import launch_bright_repair as launch

QUERY_SET = launch.reasoning.QUERY_SET
SUBSETS = launch.SUBSETS
METHOD_NAMES = {"e0": "E0", "infonce": "InfoNCE", "lambdaloss": "LambdaLoss", "reler": "RELER"}
DEFAULT_OUTPUT = launch.iclr2027.ROOT / "paper/_summary/bright_gpt4_reasoning"


def read_failures(log_dir):
    failures, warnings = {}, []
    for path in sorted(log_dir.glob("slot-*/failures.json")):
        try:
            entries = json.loads(path.read_text())
            if not isinstance(entries, list):
                raise ValueError("Expected a list of failure records")
            for entry in entries:
                failures.setdefault(entry["job"], []).append({"error": entry["error"], "source": str(path)})
        except (OSError, ValueError, KeyError, TypeError) as exc:
            warnings.append(f"Cannot read {path}: {exc}")
    return failures, warnings


def collect(jobs, log_dir):
    failures, warnings = read_failures(log_dir)
    runs = []
    for job in jobs:
        seed = int(job.row["config"]["seed"])
        methods = [method for method, stem in launch.SPECS[job.backbone].run_stems.items()
                   if job.row["run_id"] == stem + (f"-Seed{seed}" if seed != 42 else "")]
        if len(methods) != 1:
            raise ValueError(f"Cannot identify main-table method: {job.name}")
        root = launch.result_root(job)
        run = {
            "backbone": job.backbone, "method": methods[0], "seed": seed,
            "run": job.row["run_id"], "query_set": QUERY_SET,
            "status": "missing", "completed_subsets": 0, "missing_subsets": list(SUBSETS),
            "avg_ndcg_at_10": None, "scores": {}, "source": None,
            "result_root": str(root), "error": None, "failures": [],
        }
        try:
            path, payload, entries = launch.read_result(root)
            if payload is not None and payload.get("task_name", "BrightRetrieval") != "BrightRetrieval":
                raise ValueError(f"Unexpected task in {path}")
            for subset, entry in entries.items():
                if "ndcg_at_10" in entry and not math.isclose(
                    float(entry["ndcg_at_10"]), float(entry["main_score"]), rel_tol=0, abs_tol=1e-8
                ):
                    raise ValueError(f"main_score differs from ndcg_at_10: {path}/{subset}")
            run["source"] = str(path) if path else None
            run["scores"] = {subset: float(entries[subset]["main_score"]) * 100
                             for subset in SUBSETS if subset in entries}
            run["completed_subsets"] = len(entries)
            run["missing_subsets"] = [s for s in SUBSETS if s not in entries]
            if len(entries) == len(SUBSETS):
                run["status"] = "complete"
                run["avg_ndcg_at_10"] = mean(run["scores"].values())
            elif path is not None:
                run["status"] = "partial"
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            run["status"] = "invalid"
            run["error"] = str(exc)
        # A complete new result supersedes stale failure logs from an earlier attempt.
        if run["status"] != "complete" and job.name in failures:
            run["failures"] = failures[job.name]
            if run["status"] != "invalid":
                run["status"] = "failed"
        runs.append(run)

    groups = []
    for backbone in launch.SPECS:
        for method in launch.reasoning.METHODS:
            selected = [r for r in runs if r["backbone"] == backbone and r["method"] == method]
            expected = {42} if method == "e0" else set(launch.reasoning.SEEDS)
            if len(selected) != len(expected) or {r["seed"] for r in selected} != expected:
                raise ValueError(f"Unexpected/duplicate seeds: {backbone}/{method}")
            completed = [r for r in selected if r["status"] == "complete"]
            ready = len(completed) == len(expected)
            values = [r["avg_ndcg_at_10"] for r in completed]
            groups.append({
                "backbone": backbone, "method": method, "query_set": QUERY_SET,
                "status": "complete" if ready else "incomplete",
                "expected_runs": len(expected), "completed_runs": len(completed),
                "mean": mean(values) if ready else None,
                "sample_sd": stdev(values) if ready and len(values) > 1 else None,
                "subset_means": {s: mean(r["scores"][s] for r in completed) for s in SUBSETS} if ready else {},
                "subset_sample_sds": {s: stdev(r["scores"][s] for r in completed) for s in SUBSETS}
                    if ready and len(completed) > 1 else {},
            })
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "query_set": QUERY_SET, "query_config": "gpt4_reason", "query_field": "query",
        "score_units": "nDCG@10 x 100; equal-weight mean over 12 subsets; sample SD across seeds",
        "expected_runs": len(jobs), "completed_runs": sum(r["status"] == "complete" for r in runs),
        "status_counts": dict(Counter(r["status"] for r in runs)),
        "warnings": warnings, "runs": runs, "groups": groups,
    }


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(report, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    identity = ["backbone", "method", "seed", "run", "query_set"]
    run_rows, subset_rows, pending_rows = [], [], []
    for run in report["runs"]:
        row = {key: run[key] for key in identity}
        row.update({key: run[key] for key in ("status", "completed_subsets", "avg_ndcg_at_10", "source", "result_root", "error")})
        row["missing_subsets"] = ";".join(run["missing_subsets"])
        row["failure"] = " | ".join(f"{f['error']} ({f['source']})" for f in run["failures"])
        row.update({s: run["scores"].get(s) for s in SUBSETS})
        run_rows.append(row)
        if run["status"] != "complete":
            pending_rows.append(row)
        for subset, score in run["scores"].items():
            subset_rows.append({**{key: run[key] for key in identity}, "subset": subset,
                                "ndcg_at_10": score / 100, "score": score,
                                "run_status": run["status"], "source": run["source"]})
    fields = [*identity, "status", "completed_subsets", "missing_subsets", "avg_ndcg_at_10",
              *SUBSETS, "source", "result_root", "error", "failure"]
    write_csv(output_dir / "per_run_summary.csv", fields, run_rows)
    write_csv(output_dir / "pending.csv", fields, pending_rows)
    write_csv(output_dir / "subset_summary.csv",
              [*identity, "subset", "ndcg_at_10", "score", "run_status", "source"], subset_rows)
    group_fields = ["backbone", "method", "query_set", "status", "expected_runs", "completed_runs", "mean", "sample_sd"]
    group_rows = []
    for group in report["groups"]:
        row = {key: group[key] for key in group_fields}
        for subset in SUBSETS:
            row[subset + "_mean"] = group["subset_means"].get(subset)
            row[subset + "_sd"] = group["subset_sample_sds"].get(subset)
        group_rows.append(row)
    write_csv(output_dir / "grouped_summary.csv",
              [*group_fields, *(s + suffix for s in SUBSETS for suffix in ("_mean", "_sd"))], group_rows)
    (output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    def fmt(value):
        return "—" if value is None else f"{value:.2f}"

    lines = ["# BRIGHT GPT-4 query results", "",
             f"Complete: {report['completed_runs']}/{report['expected_runs']} runs. Query set: `{QUERY_SET}`.", "",
             "Scores: nDCG@10 × 100. Avg. is the unweighted mean over 12 subsets. SD is sample SD across seeds.",
             "Group statistics require every expected run and all 12 subsets. E0 is a single run; its SD is omitted.", "",
             "| Backbone | Method | Complete | Avg. | Sample SD |",
             "| --- | --- | ---: | ---: | ---: |"]
    for group in report["groups"]:
        lines.append(f"| {group['backbone']} | {METHOD_NAMES[group['method']]} | "
                     f"{group['completed_runs']}/{group['expected_runs']} | {fmt(group['mean'])} | {fmt(group['sample_sd'])} |")
    lines += ["", "## Pending / invalid runs", ""]
    if not pending_rows:
        lines.append("All 30 expected evaluations are complete.")
    for row in pending_rows:
        details = row["error"] or row["failure"] or row["missing_subsets"]
        lines.append(f"- {row['backbone']}/{row['run']}: {row['status']} ({row['completed_subsets']}/12); {details}")
    if report["warnings"]:
        lines += ["", "## Log warnings", "", *["- " + w for w in report["warnings"]]]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for backbone in launch.SPECS:
        parser.add_argument("--config-" + backbone, type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log-dir", type=Path, default=launch.iclr2027.ROOT / "logs/bright-repair")
    parser.add_argument("--require-complete", action="store_true", help="Write the report, then exit 1 unless all 30 runs are complete")
    args = parser.parse_args(argv)
    args.cost_4b = 4.0  # Only needed to reuse the launcher's exact checkpoint selection.
    report = collect(launch.make_jobs(args), args.log_dir)
    write_report(report, args.output_dir)
    print(f"Complete: {report['completed_runs']}/{report['expected_runs']}; {report['status_counts']}")
    print(args.output_dir / "summary.md")
    return int(args.require_complete and report["completed_runs"] != report["expected_runs"])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
