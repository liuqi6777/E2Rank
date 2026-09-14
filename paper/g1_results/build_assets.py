#!/usr/bin/env python3
"""Rebuild G1 report tables/figures from the checked-in, rounded result CSVs.

Run from any directory with Python + Matplotlib. No training or evaluation runs.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "paper/_summary/g1_bright"
BEST = "G1-A-MRRAlign090"
E0 = "G1-E0"
LL = "G1-J-LL-Scaled"
DOMAINS = [
    "biology", "earth_science", "economics", "psychology", "robotics",
    "stackoverflow", "sustainable_living", "pony", "leetcode", "aops",
    "theoremqa_theorems", "theoremqa_questions",
]
ALIGNMENTS = [0.40, 0.5302373892742263, 0.65, 0.80, 0.90, 0.95]
SWEEPS = {
    "Binary MRR@10": ["G1-A-MRRAlign040", "G1-A-MRR", "G1-A-MRRAlign065",
                      "G1-J-RL-MRRSmall", BEST, "G1-A-MRRAlign095"],
    "Binary nDCG@10": ["G1-A-BinaryNDCGAlign040", "G1-A-Binary", "G1-A-BinaryNDCGAlign065",
                       "G1-A-BinaryNDCGAlign080", "G1-A-BinaryNDCGAlign090", "G1-A-BinaryNDCGAlign095"],
    "Graded nDCG@10": ["G1-A-NDCGAlign040", "G1-J-RL", "G1-A-NDCGAlign065",
                       "G1-A-FixedSmall", "G1-A-NDCGAlign090", "G1-A-NDCGAlign095"],
}
GRID = {
    16: [f"G1-A-MRRG16Align{x}" for x in ("080", "090", "095")],
    32: SWEEPS["Binary MRR@10"][3:],
    64: [f"G1-A-MRRG64Align{x}" for x in ("080", "090", "095")],
}


def read_results():
    runs = list(csv.DictReader((SOURCE / "run_summary.csv").open()))
    subsets = list(csv.DictReader((SOURCE / "subset_summary.csv").open()))
    scores = {}
    domain_scores = defaultdict(dict)
    for row in runs:
        run = row["run"].removesuffix("-s42")
        if run in scores:
            raise ValueError(f"Duplicate run: {run}")
        if any(int(row[k]) != v for k, v in
               [("tasks_found", 1), ("tasks_missing", 0), ("errors", 0)]):
            raise ValueError(f"Incomplete evaluation: {run}")
        scores[run] = float(row["mean_task_score"])
    for row in subsets:
        run = row["run"].removesuffix("-s42")
        domain = row["subset"]
        if row["task"] != "BrightRetrieval" or domain in domain_scores[run]:
            raise ValueError(f"Invalid or duplicate subset: {run}/{domain}")
        domain_scores[run][domain] = float(row["score"])
    if set(scores) != set(domain_scores):
        raise ValueError("Run/subset inventories differ")
    mismatches = {}
    for run, ds in domain_scores.items():
        if set(ds) != set(DOMAINS):
            raise ValueError(f"Incomplete domains: {run}")
        # Both exports round independently to 0.01; do not overwrite the official macro.
        mismatches[run] = abs(scores[run] - sum(ds.values()) / len(DOMAINS))
        if mismatches[run] > 0.01000001:
            raise ValueError(f"Macro discrepancy exceeds rounding tolerance: {run}")
    if len(runs) != 46 or len(subsets) != 552:
        raise ValueError("This report snapshot expects 46 runs and 552 subset rows")
    return scores, domain_scores, max(mismatches.values())


def write_tables(scores, ds, mismatch):
    text = ["# G1 完整结果附表", "",
            "由 `build_assets.py` 从原始汇总 CSV 生成。分数为 BRIGHT 12 领域宏平均 nDCG@10 × 100。",
            "训练行均为 seed 42，表中省略 `-s42`；E0 为一次原始模型评测。差值由已舍入宏平均计算。", "",
            "| Run ID | 分数 | 相对 E0 | 相对最强监督 LL-Scaled |",
            "|---|---:|---:|---:|"]
    for run in sorted(scores, key=lambda r: (-scores[r], r)):
        text.append(f"| `{run}` | {scores[run]:.2f} | {scores[run]-scores[E0]:+.2f} | {scores[run]-scores[LL]:+.2f} |")
    text += ["", "## 全部配置的 BRIGHT 逐领域结果", "",
             "列顺序与源 CSV 一致，宏平均置于最后。", "",
             "| Run ID | " + " | ".join(DOMAINS) + " | Avg. |",
             "|---|" + "---:|" * 13]
    for run in sorted(scores, key=lambda r: (-scores[r], r)):
        text.append("| `" + run + "` | " + " | ".join(f"{ds[run][d]:.2f}" for d in DOMAINS)
                    + f" | {scores[run]:.2f} |")
    text += ["", "完整领域矩阵见 [all_domains.csv](all_domains.csv)。",
             "原始来源：[run_summary.csv](../_summary/g1_bright/run_summary.csv)、",
             "[subset_summary.csv](../_summary/g1_bright/subset_summary.csv)。", ""]
    (OUT / "all_runs.md").write_text("\n".join(text))
    with (OUT / "all_domains.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", *DOMAINS, "macro"])
        for run in sorted(scores, key=lambda r: (-scores[r], r)):
            writer.writerow([run, *[f"{ds[run][d]:.2f}" for d in DOMAINS], f"{scores[run]:.2f}"])
    audit = {
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(SOURCE.glob("*.csv"))},
        "runs": len(scores), "training_runs": len(scores)-1,
        "subset_rows": sum(map(len, ds.values())),
        "max_macro_rounding_discrepancy": mismatch,
        "best_run": BEST, "best_score": scores[BEST],
        "comparisons": {},
        "group_alignment_grid": {g: [scores[r] for r in runs] for g, runs in GRID.items()},
        "reward_alignment_sweeps": {name: [scores[r] for r in runs] for name, runs in SWEEPS.items()},
    }
    for ref in [E0, LL, "G1-J-CL", "G1-J-RN", "G1-A-BinaryNDCGAlign090"]:
        deltas = {d: round(ds[BEST][d]-ds[ref][d], 2) for d in DOMAINS}
        ordered = sorted(deltas, key=deltas.get, reverse=True)
        top3 = ordered[:3]
        remaining = [d for d in DOMAINS if d not in top3]
        audit["comparisons"][ref] = {
            "macro_delta": round(scores[BEST]-scores[ref], 2),
            "relative_percent": (scores[BEST]/scores[ref]-1)*100,
            "wins": sum(x > 0 for x in deltas.values()),
            "ties": sum(x == 0 for x in deltas.values()),
            "losses": sum(x < 0 for x in deltas.values()),
            "domain_deltas": deltas, "top3_gain_domains": top3,
            "top3_share_of_net_domain_gain": sum(deltas[d] for d in top3)/sum(deltas.values()),
            "remaining9_mean_delta_descriptive_only": sum(deltas[d] for d in remaining)/len(remaining),
        }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False)+"\n")


def draw_figures(scores, ds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})

    def save(fig, name):
        fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)

    runs = [E0, "G1-J-CL", "G1-J-RN", LL,
            "G1-J-LL-Binary-Scaled", "G1-A-FixedSmall",
            "G1-A-BinaryNDCGAlign090", BEST]
    labels = ["E0", "InfoNCE", "RankNet",
              "Graded LL (sigma=33.33)", "Binary LL (sigma=33.33)",
              "Graded RL (a=0.80)",
              "Binary nDCG RL (a=0.90)", "Binary MRR RL (a=0.90)"]
    fig, ax = plt.subplots(figsize=(10.4, 4.8), layout="constrained")
    colors = ["#7b8794"]+["#6792b6"]*4+["#e4a652"]*2+["#178477"]
    bars = ax.barh(labels, [scores[r] for r in runs], color=colors, height=.67)
    ax.bar_label(bars, fmt="%.2f", padding=5)
    ax.axvline(scores[E0], color="#7b8794", ls=":", lw=1)
    ax.axvline(scores[LL], color="#6792b6", ls="--", lw=1)
    ax.set(xlim=(0, 24), xlabel="BRIGHT macro nDCG@10 (0–100)",
           title="G1: supervised controls and developed RL configurations")
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=.15)
    save(fig, "main_results")

    fig, (ax, heat) = plt.subplots(1, 2, figsize=(12.8, 4.8),
                                   gridspec_kw={"width_ratios": [1.35, 1]}, layout="constrained")
    for (label, runs), color, marker in zip(SWEEPS.items(), ["#178477", "#bd692b", "#586aa3"], ["o", "s", "^"]):
        ax.plot(ALIGNMENTS, [scores[r] for r in runs], color=color, marker=marker, label=label, lw=2)
    ax.axhline(scores[LL], color="#7b8794", ls="--", label="Best supervised: 19.50", lw=1)
    ax.set(xlabel="Target alignment (larger = weaker exploration)",
           ylabel="BRIGHT macro nDCG@10 (0–100)", ylim=(16.8, 22.7),
           title="(a) Reward × exploration, G=32")
    ax.set_xticks(ALIGNMENTS, ["0.40", "0.5302", "0.65", "0.80", "0.90", "0.95"])
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    ax.grid(alpha=.17)
    matrix = np.array([[scores[r] for r in runs] for runs in GRID.values()])
    im = heat.imshow(matrix, cmap="YlGnBu", vmin=17, vmax=22.1, aspect="auto")
    for i in range(3):
        for j in range(3):
            heat.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                      color="white" if matrix[i,j] > 20 else "#102a43", fontsize=13)
    heat.set(xticks=range(3), xticklabels=["0.80", "0.90", "0.95"],
             yticks=range(3), yticklabels=["16", "32", "64"],
             xlabel="Target alignment", ylabel="Group size G",
             title="(b) Binary MRR@10 local grid")
    fig.colorbar(im, ax=heat, label="BRIGHT macro nDCG@10", shrink=.85)
    save(fig, "exploration_grid")

    fig, ax = plt.subplots(figsize=(10.5, 6.1), layout="constrained")
    y = np.arange(len(DOMAINS))
    for offset, ref, label, color in [(-.18, E0, "Selected RL − E0", "#9caab7"),
                                     (.18, LL, "Selected RL − best supervised", "#178477")]:
        values = [ds[BEST][d]-ds[ref][d] for d in DOMAINS]
        bars = ax.barh(y+offset, values, height=.32, label=label, color=color)
        ax.bar_label(bars, fmt="%+.2f", fontsize=8, padding=3)
    ax.set(yticks=y, yticklabels=DOMAINS, xlim=(-6, 22.5),
           xlabel="Difference in nDCG@10 points", title="G1: gains are broad against E0, uneven against supervised adaptation")
    ax.invert_yaxis()
    ax.axvline(0, color="#334e68", lw=.8)
    ax.grid(axis="x", alpha=.15)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    save(fig, "domain_deltas")


if __name__ == "__main__":
    scores, domain_scores, mismatch = read_results()
    write_tables(scores, domain_scores, mismatch)
    draw_figures(scores, domain_scores)
    print(f"Validated {len(scores)} runs / 552 subset rows; rebuilt G1 tables and 3 figures.")
