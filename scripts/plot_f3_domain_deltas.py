#!/usr/bin/env python3
"""Plot paired-seed BRIGHT subset differences for the 0.6B main recipe."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "paper/_summary/k0_bright/subset_summary.csv"
OUT = ROOT / "paper/analysis_results/bright_domain_deltas_k0"
METHODS = {
    "RELER": "G1-R2-RL-GradedNDCG64-CP-Align070-ShortlistUniform-K0-T1-Pairwise050",
    "InfoNCE": "G1-R2-CL-Strong",
}
SEEDS = (42, 3407, 2026)
SUBSETS = (
    ("biology", "Biology"), ("earth_science", "Earth science"),
    ("economics", "Economics"), ("psychology", "Psychology"),
    ("robotics", "Robotics"), ("stackoverflow", "Stack Overflow"),
    ("sustainable_living", "Sustainable living"),
    ("leetcode", "LeetCode"), ("pony", "Pony"),
    ("aops", "AoPS"), ("theoremqa_questions", "TheoremQA questions"),
    ("theoremqa_theorems", "TheoremQA theorems"),
)
MARKERS = {42: "o", 3407: "s", 2026: "^"}
OFFSETS = {42: -0.16, 3407: 0.0, 2026: 0.16}


def load_deltas() -> list[dict[str, object]]:
    with SOURCE.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    scores = {}
    for row in source_rows:
        if row["query_set"] != "original" or row["method"] not in METHODS.values():
            continue
        key = (row["method"], int(row["seed"]), row["subset"])
        if key in scores:
            raise ValueError(f"Duplicate BRIGHT cell: {key}")
        scores[key] = float(row["ndcg_at_10"])
    expected = {(method, seed, subset) for method in METHODS.values()
                for seed in SEEDS for subset, _ in SUBSETS}
    if set(scores) != expected:
        raise ValueError(f"Missing or extra BRIGHT cells: {expected ^ set(scores)}")

    rows = []
    for subset, label in SUBSETS:
        differences = []
        for seed in SEEDS:
            delta = 100 * (scores[METHODS["RELER"], seed, subset]
                           - scores[METHODS["InfoNCE"], seed, subset])
            differences.append(delta)
            rows.append(dict(subset=subset, label=label, seed=seed,
                             delta_ndcg_points=delta))
        rows.append(dict(subset=subset, label=label, seed="mean",
                         delta_ndcg_points=statistics.mean(differences)))
    return rows


def main() -> None:
    rows = load_deltas()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "paired_subset_deltas.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "savefig.dpi": 300,
    })
    fig, ax = plt.subplots(figsize=(6.75, 3.65))
    lookup = {(r["subset"], r["seed"]): float(r["delta_ndcg_points"]) for r in rows}
    ax.axvline(0, color="#777777", linewidth=0.8, zorder=1)
    for i, (subset, _) in enumerate(SUBSETS):
        for seed in SEEDS:
            ax.scatter(lookup[subset, seed], i + OFFSETS[seed],
                       marker=MARKERS[seed], s=22, facecolor="#b0b0b0",
                       edgecolor="#ffffff", linewidth=0.5, zorder=3)
        ax.scatter(lookup[subset, "mean"], i, marker="D", s=30,
                   facecolor="#2a78d6", edgecolor="#ffffff",
                   linewidth=0.6, zorder=4)
    for boundary in (6.5, 8.5):
        ax.axhline(boundary, color="#c9c7c0", linewidth=0.8, zorder=1)
    ax.set_yticks(range(len(SUBSETS)), [label for _, label in SUBSETS])
    ax.invert_yaxis()
    ax.set_xlim(-5, 15)
    ax.set_xticks((-5, 0, 5, 10, 15))
    ax.set_xlabel("RELER − InfoNCE nDCG@10 (points)")
    ax.grid(axis="x", color="#e1e0d9", linewidth=0.6)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], marker=MARKERS[seed], linestyle="none",
                      markersize=5, markerfacecolor="#b0b0b0",
                      markeredgecolor="#ffffff", label=f"seed {seed}")
               for seed in SEEDS]
    handles.append(Line2D([], [], marker="D", linestyle="none", markersize=5,
                          markerfacecolor="#2a78d6", markeredgecolor="#ffffff",
                          label="three-seed mean"))
    ax.legend(handles=handles, loc="lower right", ncol=2, frameon=False,
              fontsize=7, columnspacing=0.8, handletextpad=0.3)
    fig.subplots_adjust(left=0.30, right=0.985, bottom=0.12, top=0.98)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"paired_subset_deltas.{suffix}")
    plt.close(fig)
    means = [lookup[subset, "mean"] for subset, _ in SUBSETS]
    (OUT / "README.md").write_text(
        "# F3: paired-seed BRIGHT domain differences\n\n"
        "Source: `paper/_summary/k0_bright/subset_summary.csv` (original-query "
        "0.6B RELER K=0, pairwise weight 0.5, versus InfoNCE). Three paired "
        "training seeds 42/3407/2026; nDCG@10 values are multiplied by 100. "
        "`paired_subset_deltas.csv` contains all plotted per-seed points and "
        "three-seed means. Recreate with `python scripts/plot_f3_domain_deltas.py`.\n\n"
        f"The equal-subset macro difference from these four-decimal source "
        f"scores is {statistics.mean(means):.3f} points. Subsets are "
        "evaluation domains, not independent training repeats.\n"
    )


if __name__ == "__main__":
    main()
