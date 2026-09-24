#!/usr/bin/env python3
"""Export K=0 training-reward histories and build the F2 reward-curve figures.

Implements paper/ANALYSIS_PLAN.md §F2. Reads the six-run list in
paper/analysis/reward_curve_runs.csv (RELER = K=0 + pairwise 0.5 and the
graded-only control, three paired seeds each), extracts per-step reward fields
from each run's final trainer_state.json, validates step coverage, the
combined-reward identity and the resolved suite config, and writes into
paper/analysis_results/reward_curves_k0/:

  runs/<run>/history.csv        per-step logged fields (original key names)
  runs/<run>/run_manifest.json  paths, steps, config provenance, BRIGHT eval
  curves.csv                    panel/series aggregation, mean and sample SD
  curves_per_seed.csv           per-seed values behind every plotted point
  bright_final.csv              final-checkpoint BRIGHT macro averages
  reward_curves_k0_main.*       two-panel main figure (vector PDF + PNG)
  reward_curves_k0_per_seed.*   per-seed raw-lines appendix figure
  README.md                     figure caption, field mapping, file inventory

CPU only; imports no training code (field semantics are pinned by hashing the
source files that define them). Every plotted point is recomputable from
curves_per_seed.csv / history.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RUNS_CSV = ROOT / "paper/analysis/reward_curve_runs.csv"
SUITE = ROOT / "configs/experiments/iclr2027/suite_iclr2027_final_k0.yaml"
CHECKPOINT_ROOT = ROOT / "checkpoints/iclr2027-final-k0"
OUTPUT_DIR = ROOT / "paper/analysis_results/reward_curves_k0"
BRIGHT_SUBSETS = ("aops", "biology", "earth_science", "economics", "leetcode",
                  "pony", "psychology", "robotics", "stackoverflow",
                  "sustainable_living", "theoremqa_questions", "theoremqa_theorems")

# Fields exported verbatim from trainer_state.json log_history. Pure graded
# runs log only the subset without pairwise/combined keys.
HISTORY_FIELDS = ("reward/mean", "reward/pairwise/mean", "reward/combined_mean",
                  "reward/std", "reward/min", "reward/max", "train/loss",
                  "train/loss_listwise", "train/loss_pairwise", "train/loss_pairwise_weighted")
# Source files that define the logged fields; hashed into each manifest.
FIELD_SOURCE_FILES = ("src/grpo.py", "src/pairwise_projection.py", "src/shortlists.py")

COMBINED_IDENTITY_TOLERANCE = 1e-3

# Series display names (curves.csv, legends, direct labels).
PANEL_A = {"RELER": "RELER", "Listwise": "Graded-only"}
PANEL_B = (("graded", "Graded term", "reward/mean"),
           ("pairwise", "Pairwise term (0.5·r)", "reward/pairwise_weighted"),
           ("combined", "Combined", "reward/combined_mean"))

# Validated categorical palette (dataviz six checks: CVD worst adjacent ΔE 47.2;
# aqua/yellow sit below 3:1 on the light surface, so every series also carries a
# visible direct end label). Text and axes wear ink tokens, never series colors.
SURFACE, INK, INK_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, AQUA, YELLOW = "#2a78d6", "#1baf7a", "#eda100"
PANEL_A_COLORS = {"RELER": BLUE, "Listwise": AQUA}
PANEL_B_COLORS = {"graded": BLUE, "pairwise": AQUA, "combined": YELLOW}
SEED_MARKERS = {42: "o", 3407: "s", 2026: "^"}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state():
    def run(*args):
        return subprocess.check_output(["git", *args], text=True, cwd=ROOT).strip()
    return dict(commit=run("rev-parse", "HEAD"), dirty=bool(run("status", "--porcelain")))


def strip_seed_suffix(run):
    """`<recipe>-s<seed>` -> `<recipe>` (suite run keys carry no s42 suffix)."""
    parts = run.rsplit("-s", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return run


def resolve_config(run, seed, suite_path):
    """Resolve a run's overrides from the suite, following the import chain."""
    suite = yaml.safe_load((ROOT / suite_path).read_text())
    key = strip_seed_suffix(run)
    if key in suite.get("runs", {}):
        entry = suite["runs"][key]
        return entry["overrides"], [str(suite_path), f"runs:{key}"]
    import_entry = (suite.get("imports") or {}).get(key)
    if import_entry:
        source_suite_path, source_run = import_entry["suite"], import_entry["source_run"]
        source_suite = yaml.safe_load((ROOT / source_suite_path).read_text())
        entry = source_suite["runs"][source_run]
        return entry["overrides"], [str(suite_path), f"imports:{key}",
                                    source_suite_path, f"runs:{source_run}"]
    raise SystemExit(f"Run {run} not found in {suite_path} (looked up as {key})")


def load_history(run_dir, expected_steps):
    """Return per-step log rows from the final checkpoint's trainer_state."""
    trainer_state_path = run_dir / f"checkpoint-{expected_steps}/trainer_state.json"
    if not trainer_state_path.exists():
        raise SystemExit(f"Missing {trainer_state_path}")
    state = json.loads(trainer_state_path.read_text())
    history = state["log_history"]
    steps = [entry.get("step") for entry in history]
    problems = []
    if len(set(steps)) != len(steps):
        problems.append("duplicate steps in log_history")
    if steps != sorted(steps):
        problems.append("log_history not sorted by step")
    expected_range = list(range(1, expected_steps + 1))
    missing = sorted(set(expected_range) - set(steps))
    extra = sorted(set(steps) - set(expected_range))
    if missing:
        problems.append(f"missing steps {missing[:10]}{'...' if len(missing) > 10 else ''}")
    if extra:
        problems.append(f"unexpected steps {extra}")
    global_step = state.get("global_step")
    if global_step != expected_steps:
        problems.append(f"trainer global_step {global_step} != expected {expected_steps}")
    return dict(path=trainer_state_path, history=history, steps=steps,
                global_step=global_step, missing=missing, problems=problems)


def load_bright(run_dir):
    """Final-checkpoint original-query BRIGHT eval: per-subset nDCG@10 ×100."""
    hits = sorted(run_dir.glob("mteb_eval/bright/*/no_revision_available/BrightRetrieval.json"))
    if len(hits) != 1:
        raise SystemExit(f"Expected one BRIGHT eval JSON under {run_dir}/mteb_eval, found {len(hits)}")
    report = json.loads(hits[0].read_text())
    scores = {}
    for entry in report["scores"]["standard"]:
        subset = entry["hf_subset"]
        if subset in scores:
            raise SystemExit(f"Duplicate subset {subset} in {hits[0]}")
        scores[subset] = entry["ndcg_at_10"] * 100.0
    if set(scores) != set(BRIGHT_SUBSETS):
        raise SystemExit(f"Unexpected BRIGHT subsets in {hits[0]}: {sorted(scores)}")
    return dict(path=hits[0], mteb_version=report.get("mteb_version"),
                subsets=scores, average=statistics.mean(scores.values()))


def export_run(entry, checkpoint_root):
    """history.csv + run_manifest.json for one run; returns parsed rows."""
    recipe, seed, run = entry["recipe"], int(entry["seed"]), entry["run"]
    pairwise_coef = float(entry["pairwise_coef"])
    expected_steps = int(entry["expected_optimizer_steps"])
    run_dir = checkpoint_root / run
    if not run_dir.is_dir():
        raise SystemExit(f"Missing run directory {run_dir}")

    overrides, provenance = resolve_config(run, seed, SUITE.relative_to(ROOT))
    config_coef = float(overrides.get("reward_shortlist_pairwise_coef", 0.0))
    if not math.isclose(config_coef, pairwise_coef, abs_tol=1e-9):
        raise SystemExit(f"{run}: pairwise coef {pairwise_coef} (runs CSV) != "
                         f"{config_coef} (suite config)")
    if int(overrides.get("seed", seed)) != seed:
        raise SystemExit(f"{run}: suite seed {overrides.get('seed')} != runs CSV seed {seed}")

    trainer = load_history(run_dir, expected_steps)
    if trainer["problems"]:
        raise SystemExit(f"{run}: " + "; ".join(trainer["problems"]))
    exploration_path = run_dir / "exploration_state.json"
    exploration_step = None
    if exploration_path.exists():
        exploration_step = json.loads(exploration_path.read_text()).get("exploration", {}).get("step")

    # Combined-reward identity: combined == graded + coef * raw pairwise.
    identity_violations = []
    for row in trainer["history"]:
        keys = ("reward/mean", "reward/pairwise/mean", "reward/combined_mean")
        if not all(key in row for key in keys):
            continue
        violation = abs(row["reward/combined_mean"]
                        - (row["reward/mean"] + pairwise_coef * row["reward/pairwise/mean"]))
        identity_violations.append(violation)
    max_violation = max(identity_violations, default=None)
    if pairwise_coef and max_violation is None:
        raise SystemExit(f"{run}: pairwise run logged no combined-reward rows")
    if max_violation is not None and max_violation > COMBINED_IDENTITY_TOLERANCE:
        raise SystemExit(f"{run}: combined-reward identity violated by {max_violation:.2e}")

    bright = load_bright(run_dir)

    out_dir = OUTPUT_DIR / "runs" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", *HISTORY_FIELDS, "reward/pairwise_weighted"])
        for row in sorted(trainer["history"], key=lambda r: r["step"]):
            values = [row.get(field, "") for field in HISTORY_FIELDS]
            weighted = (f"{pairwise_coef * row['reward/pairwise/mean']:.10g}"
                        if "reward/pairwise/mean" in row else "")
            writer.writerow([row["step"], *(f"{v:.10g}" if isinstance(v, float) else v
                                            for v in values), weighted])

    final_weights = run_dir / "model.safetensors"
    manifest = dict(
        run=run, recipe=recipe, seed=seed, pairwise_coef=pairwise_coef,
        run_directory=str(run_dir.relative_to(ROOT)),
        trainer_state=dict(path=str(trainer["path"].relative_to(ROOT)),
                           sha256=sha256_file(trainer["path"]),
                           log_entries=len(trainer["history"]),
                           steps=[trainer["steps"][0], trainer["steps"][-1]],
                           global_step=trainer["global_step"],
                           missing_steps=trainer["missing"]),
        exploration_state=dict(path=str(exploration_path.relative_to(ROOT)),
                               step=exploration_step) if exploration_step is not None else None,
        final_weights=dict(path=str(final_weights.relative_to(ROOT)),
                           bytes=final_weights.stat().st_size,
                           mtime=time.strftime("%Y-%m-%dT%H:%M:%S",
                                               time.localtime(final_weights.stat().st_mtime))) if final_weights.exists() else None,
        config=dict(source="suite yaml (overrides verbatim)", provenance=provenance,
                    overrides=overrides),
        field_semantics=dict(
            graded="reward/mean — graded nDCG@10 training reward of own candidates "
                   "(shortlist path, K=0); the runs-CSV W&B key train/reward/mean is "
                   "the trainer-prefixed name of this field",
            pairwise_raw="reward/pairwise/mean — unweighted pairwise indicator reward",
            pairwise_weighted="reward/pairwise_weighted — derived column: coef × reward/pairwise/mean",
            combined="reward/combined_mean — graded + coef × pairwise, as logged by training",
            definition_sources={name: dict(sha256=sha256_file(ROOT / name))
                                for name in FIELD_SOURCE_FILES}),
        combined_identity=dict(checks=len(identity_violations),
                               max_abs_violation=max_violation,
                               tolerance=COMBINED_IDENTITY_TOLERANCE),
        bright_final=dict(eval_json=str(bright["path"].relative_to(ROOT)),
                          mteb_version=bright["mteb_version"],
                          average_ndcg_at_10_x100=round(bright["average"], 4),
                          subsets={k: round(v, 4) for k, v in sorted(bright["subsets"].items())}),
        wandb_run_id=None,
        wandb_note="no W&B artifacts were copied with these runs; the local "
                   "trainer_state.json (hashed above) is the stepwise authority",
    )
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    rows = []
    for row in trainer["history"]:
        record = {"step": row["step"]}
        for field in HISTORY_FIELDS:
            record[field] = row.get(field)
        if "reward/pairwise/mean" in row:
            record["reward/pairwise_weighted"] = pairwise_coef * row["reward/pairwise/mean"]
        rows.append(record)
    return dict(entry=entry, rows=rows, bright=bright, manifest=manifest)


def aggregate(exports):
    """curves.csv (mean ± sample SD across seeds) and curves_per_seed.csv."""
    by_recipe = {}
    for export in exports:
        by_recipe.setdefault(export["entry"]["recipe"], []).append(export)

    panel_rows, seed_rows = [], []
    for recipe, group in by_recipe.items():
        series_name = PANEL_A[recipe]
        per_seed = {int(export["entry"]["seed"]): {row["step"]: row["reward/mean"]
                                                   for row in export["rows"]}
                    for export in group}
        steps = sorted({step for values in per_seed.values() for step in values})
        for step in steps:
            values = [per_seed[seed][step] for seed in per_seed if step in per_seed[seed]]
            panel_rows.append(["A", series_name, step, f"{statistics.mean(values):.10g}",
                               f"{statistics.stdev(values):.10g}" if len(values) > 1 else "",
                               len(values)])
            for seed, values in per_seed.items():
                if step in values:
                    seed_rows.append(["A", series_name, recipe, seed,
                                      next(e["entry"]["run"] for e in group
                                           if int(e["entry"]["seed"]) == seed),
                                      step, f"{values[step]:.10g}"])

    # Panel B: RELER components only.
    for export in by_recipe.get("RELER", []):
        seed = export["entry"]["seed"]
        for row in export["rows"]:
            for key, series_name, field in PANEL_B:
                if row.get(field) is None:
                    continue
                seed_rows.append(["B", series_name, "RELER", seed,
                                  export["entry"]["run"], row["step"], f"{row[field]:.10g}"])
    panel_b_fields = {key: field for key, _, field in PANEL_B}
    for key, series_name, _ in PANEL_B:
        field = panel_b_fields[key]
        per_seed = {int(export["entry"]["seed"]): {row["step"]: row[field] for row in export["rows"]
                                                   if row.get(field) is not None}
                    for export in by_recipe.get("RELER", [])}
        steps = sorted({step for values in per_seed.values() for step in values})
        for step in steps:
            values = [per_seed[seed][step] for seed in per_seed if step in per_seed[seed]]
            if not values:
                continue
            panel_rows.append(["B", series_name, step, f"{statistics.mean(values):.10g}",
                               f"{statistics.stdev(values):.10g}" if len(values) > 1 else "",
                               len(values)])

    panel_rows.sort(key=lambda r: (r[0], r[1], int(r[2])))
    seed_rows.sort(key=lambda r: (r[0], r[1], int(r[3]), int(r[5])))
    with (OUTPUT_DIR / "curves.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel", "series", "step", "mean", "sample_sd", "n_seeds"])
        writer.writerows(panel_rows)
    with (OUTPUT_DIR / "curves_per_seed.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["panel", "series", "recipe", "seed", "run", "step", "value"])
        writer.writerows(seed_rows)
    return panel_rows, seed_rows


def read_curves(output_dir):
    """Read back the written CSVs so figures plot exactly the shipped data."""
    with (output_dir / "curves.csv").open() as handle:
        panel = [row for row in csv.DictReader(handle)]
    with (output_dir / "curves_per_seed.csv").open() as handle:
        per_seed = [row for row in csv.DictReader(handle)]
    return panel, per_seed


def configure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.titlesize": 8.5, "axes.titlecolor": INK, "axes.titlepad": 6,
        "axes.labelsize": 8, "axes.labelcolor": INK_2,
        "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "xtick.minor.size": 0, "ytick.minor.size": 0,
        "legend.frameon": False, "legend.fontsize": 7,
        "text.color": INK,
        "figure.dpi": 300, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    return plt


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_xlim(0, 113)
    ax.set_xticks([0, 25, 50, 75, 100])


def add_series(ax, steps, mean, sd, color, label):
    ax.fill_between(steps, [m - s for m, s in zip(mean, sd)],
                    [m + s for m, s in zip(mean, sd)], color=color, alpha=0.12,
                    linewidth=0, zorder=2)
    ax.plot(steps, mean, color=color, linewidth=1.8, solid_capstyle="round",
            solid_joinstyle="round", label=label, zorder=3)


def direct_end_labels(ax, series, min_gap_fraction=0.06):
    """Right-edge labels in ink text beside a colored end dot; skip on collision."""
    values = [data["mean"][-1] for _, data in series]
    low, high = ax.get_ylim()
    if len(values) > 1 and min(b - a for a, b in zip(sorted(values), sorted(values)[1:])) \
            < min_gap_fraction * (high - low):
        return False
    for label, data in series:
        ax.plot([data["steps"][-1]], [data["mean"][-1]], marker="o", markersize=4.5,
                color=data["color"], markeredgecolor=SURFACE, markeredgewidth=1.1,
                zorder=4, clip_on=False)
        ax.annotate(label, (data["steps"][-1], data["mean"][-1]),
                    xytext=(6, 0), textcoords="offset points",
                    fontsize=7, color=INK_2, va="center", ha="left",
                    annotation_clip=False)
    ax.set_xlim(0, 113 * 1.22)  # label gutter; ticks stop at 100
    return True


def nice_ylim(ax):
    low, high = ax.get_ylim()
    span = high - low
    ax.set_ylim(math.floor((low - 0.08 * span) * 20) / 20,
                math.ceil((high + 0.12 * span) * 20) / 20)


def panel_series(panel_rows, panel, name):
    steps, mean, sd = [], [], []
    for row in panel_rows:
        if row["panel"] == panel and row["series"] == name:
            steps.append(int(row["step"]))
            mean.append(float(row["mean"]))
            sd.append(float(row["sample_sd"] or 0.0))
    return steps, mean, sd


def plot_main(panel_rows, output_dir):
    plt = configure_matplotlib()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.75, 2.55))

    # Panel A: same-definition graded reward, both groups.
    series_a = []
    for recipe, display in PANEL_A.items():
        steps, mean, sd = panel_series(panel_rows, "A", display)
        if not steps:
            continue
        add_series(ax_a, steps, mean, sd, PANEL_A_COLORS[recipe], display)
        series_a.append((display, dict(steps=steps, mean=mean, color=PANEL_A_COLORS[recipe])))
    style_axis(ax_a)
    nice_ylim(ax_a)
    direct_end_labels(ax_a, series_a)
    ax_a.set_xlabel("Optimizer step")
    ax_a.set_ylabel("Graded nDCG@10 reward")
    ax_a.set_title("(a) Graded training reward", loc="left")
    ax_a.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), handlelength=1.6)

    # Panel B: RELER components.
    series_b = []
    for key, display, _ in PANEL_B:
        steps, mean, sd = panel_series(panel_rows, "B", display)
        if not steps:
            continue
        add_series(ax_b, steps, mean, sd, PANEL_B_COLORS[key], display)
        series_b.append((display, dict(steps=steps, mean=mean, color=PANEL_B_COLORS[key])))
    style_axis(ax_b)
    nice_ylim(ax_b)
    direct_end_labels(ax_b, series_b)
    ax_b.set_xlabel("Optimizer step")
    ax_b.set_ylabel("Training reward (RELER)")
    ax_b.set_title("(b) RELER reward components", loc="left")
    ax_b.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), handlelength=1.6)

    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.15, top=0.88, wspace=0.28)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"reward_curves_k0_main.{suffix}")
    plt.close(fig)


def plot_per_seed(per_seed_rows, output_dir):
    plt = configure_matplotlib()
    fig = plt.figure(figsize=(6.75, 4.7))
    grid = fig.add_gridspec(2, 3, height_ratios=(1.25, 1.0), hspace=0.55, wspace=0.32,
                            left=0.075, right=0.985, top=0.92, bottom=0.09)

    ax_a = fig.add_subplot(grid[0, :])
    grouped = {}
    for row in per_seed_rows:
        if row["panel"] == "A":
            grouped.setdefault((row["series"], int(row["seed"])), []).append(
                (int(row["step"]), float(row["value"])))
    recipe_of = {display: recipe for recipe, display in PANEL_A.items()}
    group_end_means = {}
    for (series, seed), points in sorted(grouped.items()):
        points.sort()
        color = PANEL_A_COLORS[recipe_of[series]]
        ax_a.plot([p[0] for p in points], [p[1] for p in points], color=color,
                  linewidth=1.0, marker=SEED_MARKERS[seed], markersize=3.0,
                  markeredgecolor=SURFACE, markeredgewidth=0.6, markevery=14,
                  solid_capstyle="round")
        group_end_means.setdefault(series, []).append(points[-1][1])
    style_axis(ax_a)
    low, high = ax_a.get_ylim()
    span = high - low
    ax_a.set_ylim(low - 0.08 * span, high + 0.12 * span)
    end_series = []
    for series, ends in group_end_means.items():
        end_series.append((series, dict(steps=[113], mean=[statistics.mean(ends)],
                                        color=PANEL_A_COLORS[recipe_of[series]])))
    direct_end_labels(ax_a, end_series)
    # Seed identity rides on markers; the marker key sits in the empty lower right.
    handles = [plt.Line2D([], [], color=MUTED, linestyle="none", marker=marker,
                          markersize=3.5, markeredgecolor=SURFACE, markeredgewidth=0.6,
                          label=f"s{seed}") for seed, marker in SEED_MARKERS.items()]
    ax_a.legend(handles=handles, loc="lower right", ncols=3, handlelength=1.2,
                columnspacing=1.2, borderaxespad=0.4)
    ax_a.set_xlabel("Optimizer step")
    ax_a.set_ylabel("Graded nDCG@10 reward")
    ax_a.set_title("(a) Graded training reward, per seed", loc="left")

    for column, (key, display, _) in enumerate(PANEL_B):
        ax = fig.add_subplot(grid[1, column])
        grouped = {}
        for row in per_seed_rows:
            if row["panel"] == "B" and row["series"] == display:
                grouped.setdefault(int(row["seed"]), []).append(
                    (int(row["step"]), float(row["value"])))
        for seed, points in sorted(grouped.items()):
            points.sort()
            ax.plot([p[0] for p in points], [p[1] for p in points],
                    color=PANEL_B_COLORS[key], linewidth=1.0,
                    marker=SEED_MARKERS[seed], markersize=3.0,
                    markeredgecolor=SURFACE, markeredgewidth=0.6, markevery=14,
                    label=f"s{seed}", solid_capstyle="round")
        style_axis(ax)
        ax.set_xlabel("Optimizer step")
        ax.set_title(("(b) " if column == 0 else "") + display, loc="left")
        if column == 0:
            ax.set_ylabel("Training reward (RELER)")
            ax.legend(loc="lower right", handlelength=1.2, borderaxespad=0.4)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"reward_curves_k0_per_seed.{suffix}")
    plt.close(fig)


def write_readme(exports, output_dir, git):
    by_recipe = {}
    for export in exports:
        by_recipe.setdefault(export["entry"]["recipe"], []).append(export)
    bright = {recipe: [export["bright"]["average"] for export in group]
              for recipe, group in by_recipe.items()}
    bright_line = " · ".join(
        f"{PANEL_A[recipe]} {statistics.mean(values):.2f} ± {statistics.stdev(values):.2f}"
        for recipe, values in sorted(bright.items()))
    rounded_sd = {PANEL_A[recipe]: statistics.stdev([round(v, 2) for v in values])
                  for recipe, values in bright.items()}
    identity = max((export["manifest"]["combined_identity"]["max_abs_violation"] or 0.0,
                    export["entry"]["run"]) for export in exports)
    steps = sorted({row["step"] for export in exports for row in export["rows"]})
    readme = f"""# F2：K=0 训练 reward 曲线

[分析计划 §F2](../../ANALYSIS_PLAN.md) 的交付目录。生成命令（仓库根目录）：
`python scripts/export_reward_curves_k0.py`；生成于 {time.strftime("%Y-%m-%d %H:%M")}，git `{git['commit']}`{' (dirty)' if git['dirty'] else ''}。

## 图

- `reward_curves_k0_main.pdf` / `.png` — 正文 F2。面板 A：RELER（`K=0` + pairwise `λ=0.5`）与纯 graded 对照各三 seed 的**同定义** graded nDCG@10 训练 reward（optimizer step 1–{steps[-1]}）；面板 B：RELER 的 graded 项、`0.5×pairwise` 项（加权后数值）与 combined reward。带为逐 step 三 seed 样本 SD（**不是** rollout 内 `reward/std`）。
- `reward_curves_k0_per_seed.pdf` / `.png` — 附录单 seed 原线；颜色 = 组/分项（与主图一致，线端直接标组名），seed 身份由 marker 编码（○ s42、□ s3407、△ s2026）。

## 拟用图注要点

- 协议：Qwen3-Embedding-0.6B 全参数微调，shortlist 路径 `K=0`（仅自有候选，`T=1`），graded nDCG@10（teacher grades）+ pairwise `λ=0.5`，RLOO+CMP（`conditional_projection`），`G=64`，`ρ=0.70`，113 optimizer steps，global batch 128，paired seeds 42/3407/2026；两组只有 pairwise 系数不同。
- 每个 step 的训练 batch 不同：曲线是训练目标值的轨迹，不是固定验证集学习曲线，也不直接代表 BRIGHT 泛化。
- 面板 B 纵轴为 reward 原值：graded 项与 pairwise 项同为 [0,1] 尺度但语义不同（nDCG vs 配对指示 reward）；combined = graded + 0.5×pairwise。
- 相同三 seed 最终 checkpoint 的 original-query BRIGHT 12-subset 宏平均（×100）：{bright_line}。SD 为全精度逐 seed 值的样本标准差；若与正文表对数，注意结果表的 ± 来自两位小数逐 seed 值（本组即 {rounded_sd['Graded-only']:.2f}/{rounded_sd['RELER']:.2f}），与全精度在舍入边缘可能差 0.01。

## 数据核对

- 步数完整：六条运行 log_history 均为 step 1–{steps[-1]} 无缺步；`trainer_state.global_step` = 113，`exploration_state.step` = 113。
- combined 恒等式 `combined = graded + 0.5×pairwise`：全部成对 step 成立，最大绝对偏差 {identity[0]:.2e}（{identity[1]}）。
- 字段映射：W&B 键 `train/reward/mean` 等对应本地 `trainer_state.json` 的 `reward/mean` 等（trainer 加 `train/` 前缀）；语义以 `src/grpo.py`、`src/pairwise_projection.py` 的当前版本为准（sha256 见各 `run_manifest.json`）。这些运行无本地 W&B 工件，本地 `trainer_state.json` 为逐步数据权威来源。
- 纯 graded 组为从 `suite_g1_shortlist_small_k.yaml` 导入的已完成运行，RELER 组为新 suite 训练；两组配置除 `reward_shortlist_pairwise_coef` 外逐字段一致（见 manifest）。

## 文件

- `runs/<run>/history.csv`、`runs/<run>/run_manifest.json` — 逐步原始字段与来源核对。
- `curves.csv` — 面板/系列聚合（mean、sample SD、n_seeds）；`curves_per_seed.csv` — 图中每个点的原值，可复算全图。
- `bright_final.csv` — 六条运行最终 checkpoint 的 BRIGHT 12-subset nDCG@10（×100）宏平均。
- 图中不补缺步、不做平滑；如后续加 5-step trailing mean 仅作显示层并另行保留原始图。
"""
    (output_dir / "README.md").write_text(readme)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs-csv", type=Path, default=RUNS_CSV)
    parser.add_argument("--suite", type=Path, default=SUITE)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main():
    global SUITE, CHECKPOINT_ROOT, OUTPUT_DIR
    args = parse_args()
    SUITE, CHECKPOINT_ROOT, OUTPUT_DIR = args.suite, args.checkpoint_root, args.output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    entries = list(csv.DictReader(args.runs_csv.open()))
    if not entries:
        raise SystemExit(f"No runs listed in {args.runs_csv}")
    exports = [export_run(entry, CHECKPOINT_ROOT) for entry in entries]

    aggregate(exports)
    with (OUTPUT_DIR / "bright_final.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["recipe", "seed", "run", "bright_avg_ndcg_at_10_x100"])
        for export in exports:
            writer.writerow([export["entry"]["recipe"], export["entry"]["seed"],
                             export["entry"]["run"],
                             f"{export['bright']['average']:.4f}"])

    panel_rows, per_seed_rows = read_curves(OUTPUT_DIR)
    plot_main(panel_rows, OUTPUT_DIR)
    plot_per_seed(per_seed_rows, OUTPUT_DIR)
    write_readme(exports, OUTPUT_DIR, git_state())
    print(f"Wrote {len(exports)} run exports, figures and README to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
