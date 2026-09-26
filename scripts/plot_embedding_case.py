#!/usr/bin/env python3
"""Plot the four-domain, single-seed BRIGHT retrieval case from report CSVs."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
REPORT = (ROOT / "paper/analysis_results/iclr2027_representatives_k0_s3407"
          / "reports/0cbf2c8d82934128")
OUT = ROOT / "paper/analysis_results/iclr2027_representatives_k0_s3407"
DOMAINS = ("biology", "earth_science", "robotics", "theoremqa_theorems")
BINS = ("1", "2-10", "11-100", "101+")
COLORS = {"RELER": "#2a78d6", "Listwise": "#1baf7a", "InfoNCE": "#b27a00"}


def read_csv(name: str) -> list[dict[str, str]]:
    with (REPORT / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    retrieval = read_csv("retrieval.csv")
    rank_bins = read_csv("rank_bins.csv")
    scores = {(r["subset"], r["model"]): float(r["ndcg"]) for r in retrieval}
    if len(scores) != len(retrieval) or set(s for s, _ in scores) != set(DOMAINS):
        raise ValueError("Unexpected or duplicated retrieval rows")
    for domain in DOMAINS:
        for bin_name in BINS:
            values = {int(r["queries"]) for r in rank_bins
                      if r["subset"] == domain and r["reference_rank_bin"] == bin_name}
            if len(values) != 1:
                raise ValueError(f"Rank-bin query count mismatch: {domain}, {bin_name}")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
        "axes.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "savefig.dpi": 300,
    })
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.75, 2.55))
    left.axvline(0, color="#999999", linewidth=0.8)
    offsets = {"InfoNCE": -0.10, "Listwise": 0.10}
    for model in ("InfoNCE", "Listwise"):
        delta = [scores[domain, "RELER"] - scores[domain, model] for domain in DOMAINS]
        left.scatter(delta, [i + offsets[model] for i in range(len(DOMAINS))],
                     s=26, color=COLORS[model], label=f"RELER - {model}", zorder=3)
    domain_names = ("Biology", "Earth science", "Robotics", "TheoremQA Th.")
    domain_counts = {r["subset"]: int(r["queries"]) for r in retrieval if r["model"] == "E0"}
    left.set_yticks(range(len(DOMAINS)),
                    [f"{name} ({domain_counts[domain]})"
                     for domain, name in zip(DOMAINS, domain_names)])
    left.invert_yaxis()
    left.set_xlim(-0.055, 0.15)
    left.set_xticks((-0.05, 0, 0.05, 0.10, 0.15))
    left.set_xlabel("Difference in nDCG@10")
    left.set_title("(a) Domain differences", loc="left")
    left.legend(loc="lower right", frameon=False, fontsize=6.6,
                handletextpad=0.3, borderaxespad=0.2)
    left.grid(axis="x", color="#e1e0d9", linewidth=0.6)
    left.set_axisbelow(True)

    totals = {b: sum(next(int(r["queries"]) for r in rank_bins
                          if r["subset"] == domain and r["model"] == "E0"
                          and r["reference_rank_bin"] == b) for domain in DOMAINS)
              for b in BINS}
    offsets = {"InfoNCE": -0.20, "Listwise": 0.0, "RELER": 0.20}
    for model in ("InfoNCE", "Listwise", "RELER"):
        values = [statistics.mean(float(r["delta_ndcg"]) for r in rank_bins
                                  if r["model"] == model and r["reference_rank_bin"] == b)
                  for b in BINS]
        right.plot([i + offsets[model] for i in range(len(BINS))], values, "o",
                   markersize=4.6, color=COLORS[model], label=model)
    right.axhline(0, color="#999999", linewidth=0.8)
    right.set_xticks(range(len(BINS)), [f"{b}\n({totals[b]})" for b in BINS])
    right.set_xlim(-0.45, 3.45)
    right.set_ylim(-0.22, 0.35)
    right.set_ylabel("Mean nDCG@10 change\nfrom Q3E-0.6B")
    right.set_xlabel("Q3E-0.6B best-positive rank (query count)")
    right.set_title("(b) Fixed-query rank bins", loc="left")
    right.legend(loc="upper left", frameon=False, fontsize=6.6,
                 ncol=3, handletextpad=0.2, columnspacing=0.8)
    right.grid(axis="y", color="#e1e0d9", linewidth=0.6)
    right.set_axisbelow(True)

    fig.subplots_adjust(left=0.19, right=0.99, bottom=0.24, top=0.91, wspace=0.49)
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"embedding_case_s3407.{suffix}")
    plt.close(fig)


if __name__ == "__main__":
    main()
