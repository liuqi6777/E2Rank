#!/usr/bin/env python3
"""Render gradient, reward, and hyperparameter panels from audited exports."""

import csv
import json
from pathlib import Path

from analyze_f1_probe import cloud_panel
from export_reward_curves_k0 import (
    AQUA, BLUE, YELLOW, add_series, configure_matplotlib,
    panel_series, style_axis,
)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "paper/analysis_results"


def main():
    probe_dir = RESULTS / "gradient_probe_k0_pairwise"
    stem = "gradient_cloud_e0_micro100"
    with (probe_dir / f"{stem}.csv").open(newline="") as handle:
        cloud = list(csv.DictReader(handle))
    meta = json.loads((probe_dir / f"{stem}_meta.json").read_text())
    geometry = dict(
        cp_mean=meta["cmp_mean"], rloo_mean=meta["rloo_mean"],
        half_span=meta["inset_half_span"], zoom=meta["inset_view_zoom"],
    )
    for estimator, suffix in (("cmp", "cp"), ("rloo", "rloo")):
        rows = sorted((row for row in cloud if row["estimator"] == estimator),
                      key=lambda row: int(row["draw"]))
        if len(rows) != meta["draws"]:
            raise ValueError("Gradient draw count differs from exported metadata")
        geometry[f"x_{suffix}"] = [float(row["x_signal_axis"]) for row in rows]
        geometry[f"y_{suffix}"] = [float(row["y_noise_axis"]) for row in rows]
    geometry["limit"] = 1.08 * max(abs(value) for key in
                                     ("x_cp", "y_cp", "x_rloo", "y_rloo")
                                     for value in geometry[key])
    with (RESULTS / "reward_curves_k0/curves.csv").open(newline="") as handle:
        rewards = list(csv.DictReader(handle))
    with (ROOT / "paper/_summary/k0_bright/k0_bright_grouped.csv").open(newline="") as handle:
        scores = {row["method"]: row for row in csv.DictReader(handle)}
    recipe = "G1-R2-RL-GradedNDCG64-CP-Align070-ShortlistUniform-K0-T1"
    weight_scores = [(weight, scores[recipe + suffix]) for weight, suffix in
                     ((0, ""), (0.5, "-Pairwise050"), (1, "-Pairwise100"))]
    alignment_scores = [(rho, scores[recipe.replace("Align070", f"Align{round(rho * 100):03d}")
                                    + "-Pairwise050"]) for rho in (0.6, 0.7, 0.8, 0.9)]

    plt = configure_matplotlib()
    fig = plt.figure(figsize=(6.75, 2.5))
    # Give the gradient/reward panels more width and keep common vertical bounds.
    bottom, top = 0.22, 0.88
    height = top - bottom
    gradient = fig.add_axes([0.065, bottom, 0.31, height])
    reward = fig.add_axes([0.46, bottom, 0.265, height])
    small_height = (height - 0.15) / 2
    sensitivity = fig.add_axes([0.80, top - small_height, 0.185, small_height])
    alignment = fig.add_axes([0.80, bottom, 0.185, small_height])
    cloud_panel(gradient, geometry, "(a) Gradient noise")
    # Widen the x view, rather than stretching distances or shrinking the axes.
    half_x = geometry["limit"] * (0.31 * fig.get_figwidth()) / (height * fig.get_figheight())
    gradient.set_xlim(-half_x, half_x)
    gradient.set_xlabel("Projection axis 1")
    gradient.set_ylabel("Projection axis 2")
    gradient.set_xticks([-5, 0, 5])
    handles, _ = gradient.get_legend_handles_labels()
    gradient.legend(handles, ["RLOO", "RLOO+CMP"], loc="upper left", fontsize=6.5, handlelength=1.0,
                    borderaxespad=0.1)
    # The main-text inset only needs to reveal the CMP cloud; mean comparisons
    # remain in the numerical diagnostics rather than tiny overlapping labels.
    for annotation in list(gradient.texts):
        annotation.remove()
    inset = gradient.child_axes[0]
    for artist in list(inset.texts) + list(inset.lines):
        artist.remove()
    # Label components explicitly: reward/mean is nDCG, not RELER's total reward.
    reward_series = (
        ("A", "RELER", r"Row $\mathit{(a)}$: nDCG", BLUE, "-"),
        ("A", "Graded-only", r"Row $\mathit{(c)}$: nDCG", AQUA, "-"),
        ("B", "Pairwise term (0.5·r)", r"Row $\mathit{(a)}$: 0.5×pairwise", YELLOW, "-"),
    )
    for panel, name, label, color, linestyle in reward_series:
        steps, mean, sd = panel_series(rewards, panel, name)
        if steps != list(range(1, 114)):
            raise ValueError(f"Incomplete reward export for {name}")
        if any(int(row["n_seeds"]) != 3 for row in rewards
               if row["panel"] == panel and row["series"] == name):
            raise ValueError(f"Expected three seeds for {name}")
        add_series(reward, steps, mean, sd, color, label)
        reward.lines[-1].set_linestyle(linestyle)
    style_axis(reward)
    # Reserve space above the data for three readable component labels.
    reward.set_ylim(0.25, 1.15)
    reward.set_yticks([0.3, 0.5, 0.7, 0.9])
    reward.set_xticks([0, 50, 100])
    reward.set_xlabel("Optimizer step")
    reward.set_ylabel("Reward component")
    reward.set_title("(b) Reward components", loc="left")
    reward.legend(loc="upper left", handlelength=1.3, fontsize=6.5,
                  handletextpad=0.4, borderaxespad=0.1, labelspacing=0.4)
    for weight, row in weight_scores:
        if int(row["n_seeds"]) != 3:
            raise ValueError(f"Expected three seeds for pairwise weight {weight}")
        mean, sd = float(row["orig_n12_mean"]), float(row["orig_n12_sd"])
        color = BLUE if weight == 0.5 else AQUA
        sensitivity.errorbar(weight, mean, yerr=sd, fmt="o", color=color,
                             markersize=5, capsize=3, elinewidth=1.3,
                             markerfacecolor=color if weight == 0.5 else "white",
                             markeredgewidth=1.3, zorder=3)
    sensitivity.set_title(r"(c) Pairwise weight $\lambda$", loc="left")
    sensitivity.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    sensitivity.set_xlim(-0.2, 1.2)
    sensitivity.set_ylim(22.2, 23.8)
    sensitivity.set_yticks([22.5, 23, 23.5])
    sensitivity.set_axisbelow(True)
    sensitivity.grid(axis="y", color="#e1e0d9", linewidth=0.6)
    for rho, row in alignment_scores:
        if int(row["n_seeds"]) != 3:
            raise ValueError(f"Expected three seeds for alignment {rho}")
        mean, sd = float(row["orig_n12_mean"]), float(row["orig_n12_sd"])
        color = BLUE if rho == 0.7 else AQUA
        alignment.errorbar(rho, mean, yerr=sd, fmt="o", color=color,
                           markersize=5, capsize=3, elinewidth=1.3,
                           markerfacecolor=color if rho == 0.7 else "white",
                           markeredgewidth=1.3, zorder=3)
    alignment.set_title("(d) Alignment", loc="left")
    alignment.set_xlabel(r"Alignment $\rho$")
    alignment.set_xticks([0.6, 0.7, 0.8, 0.9])
    alignment.set_xlim(0.56, 0.94)
    # Match the weight panel's scale so the relative sensitivity is visible.
    for ax in (sensitivity, alignment):
        ax.set_ylim(22.2, 23.9)
        ax.set_yticks([22.5, 23.5])
    alignment.set_axisbelow(True)
    alignment.grid(axis="y", color="#e1e0d9", linewidth=0.6)
    # Explicit figure coordinates keep all three bottom labels on one baseline.
    for ax in (gradient, reward, alignment):
        ax.xaxis.set_label_coords(ax.get_position().x0 + ax.get_position().width / 2,
                                 0.105, transform=fig.transFigure)
    output = RESULTS / "training_analysis"
    output.mkdir(exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"training_analysis.{suffix}")
    plt.close(fig)
    print(output / "training_analysis.pdf")


if __name__ == "__main__":
    main()
