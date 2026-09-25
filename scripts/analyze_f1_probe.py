#!/usr/bin/env python3
"""Aggregate the F1 paired fixed-state gradient-probe JSONs into paper data.

Implements the F1 deliverable of paper/ANALYSIS_PLAN.md. Reads the three probe
reports under outputs/f1_k0_probe/ (fixed states E0 / step 50 / step 113 of the
0.6B main recipe seed 3407; three training microbatches at positions 0/100/200;
64 paired rollout draws each; conditional projection vs the unprojected RLOO
control on the identical graded + 0.5*pairwise objective) and writes
paper/analysis_results/gradient_probe_k0_pairwise/:

  probes.csv                        per-probe statistics (one figure point per row)
  gradient_cloud_<state>_micro<mb>.csv + _meta.json
                                      the data behind panel (a): one row per draw
                                      per estimator (normalized signal/noise-axis
                                      coordinates plus the raw dots on the two
                                      selected axes), and the derived quantities
                                      (axis selection, projected-signal scale,
                                      means, CMP sd, inset window and zoom)
  snr_probes.csv                    the nine panel-(b) points (snr_cp, snr_rloo)
  gradient_probe_k0_pairwise.pdf/png  two-panel figure: A, the per-draw gradient
                                      cloud of one representative probe (fixed
                                      random-axis coordinates of all 64 draws,
                                      both estimators on the same draws, thin
                                      connectors pairing them, CMP zoom inset);
                                      B, the same probes' per-step signal-to-noise.
                                      Falls back to the older noise + SNR panels
                                      when the projection readout is missing.
  README.md                         protocol, headline numbers, caption draft

Component (graded/pairwise) statistics and the mean-gradient resolution flags
ride along in probes.csv for the appendix; the main figure uses totals only.
CPU only; no training code is imported.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE_ROOT = ROOT / "outputs/f1_k0_probe"
OUTPUT_DIR = ROOT / "paper/analysis_results/gradient_probe_k0_pairwise"
STATES = (("e0", "E0"), ("step50", "step 50"), ("step113", "step 113"))
RUN_STEM = "pairwise050_seed3407_{state}_micro0-100-200.json"
PROJ_STEM = "pairwise050_seed3407_{state}_micro0-100-200_proj.json"
PAIR_KEYS = ("action_sha256", "shortlist_sha256", "graded_reward_sha256",
             "pairwise_input_sha256")

# Validated categorical palette (dataviz six checks; blue/aqua adjacent CVD
# ΔE 47.2). Identity text in ink tokens, never the series color.
SURFACE, INK, INK_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, AQUA = "#2a78d6", "#1baf7a"


def geometric_mean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def probe_rows():
    rows = []
    for state, _ in STATES:
        report = json.loads((PROBE_ROOT / RUN_STEM.format(state=state)).read_text())
        for probe in report["probes"]:
            summary = probe["summary"]
            cp, rloo = summary["conditional_projection"], summary["score_function_rloo"]
            ratio = summary["variance_ratio"]
            time_ratio = (ratio * cp["mean_forward_backward_seconds"]
                          / rloo["mean_forward_backward_seconds"])
            resolved = summary["paired_mean_difference_squared_unbiased"]
            row = dict(
                state=state,
                microbatch=probe["batches"][0]["sampler_microbatch_index"],
                sources=";".join(sorted({b for batch in probe["batches"]
                                         for b in batch["sources"]}))[:80],
                draws=len(report["rollout_seeds"]),
                variance_ratio=ratio,
                variance_time_ratio=time_ratio,
                noise_variance_cp=cp["noise_variance"],
                noise_variance_rloo=rloo["noise_variance"],
                noise_rms_cp=cp["noise_rms"],
                noise_rms_rloo=rloo["noise_rms"],
                mean_gradient_norm_cp=cp["mean_gradient_norm"],
                mean_gradient_norm_rloo=rloo["mean_gradient_norm"],
                signal_squared_cp=cp["signal_squared_unbiased"],
                signal_squared_rloo=rloo["signal_squared_unbiased"],
                forward_backward_seconds_cp=cp["mean_forward_backward_seconds"],
                forward_backward_seconds_rloo=rloo["mean_forward_backward_seconds"],
                paired_mean_difference_squared_unbiased=resolved,
                mean_difference_resolved=bool(resolved is not None and resolved > 0),
                bias_fraction_of_signal=((resolved / cp["signal_squared_unbiased"])
                                         if resolved is not None and resolved > 0
                                         and cp["signal_squared_unbiased"] > 0 else 0.0),
                sample_mean_cosine=summary["sample_mean_cosine"],
                # Per-draw signal-to-noise: ||true gradient|| estimate over the
                # per-step noise RMS; the figure's panel (b).
                snr_cp=math.sqrt(max(cp["signal_squared_unbiased"], 0.0)) / cp["noise_rms"],
                snr_rloo=math.sqrt(max(rloo["signal_squared_unbiased"], 0.0)) / rloo["noise_rms"],
                pairwise_reward_agreement_gap=probe.get("pairwise_reward_agreement_max_gap"),
                git_commit=report["git_commit"],
                diagnostic_sha256=report["diagnostic_sha256"],
            )
            for prefix, stats in (("cp", cp), ("rloo", rloo)):
                components = stats.get("components") or {}
                for key in ("graded_norm_mean", "pairwise_norm_mean",
                            "component_cosine_mean", "residual_relative_max"):
                    row[f"{prefix}_{key}"] = components.get(key)
            rows.append(row)
    return rows


def load_projection(cloud_state, cloud_microbatch, proj_report):
    """Load the per-draw projection readout, verifying it replays the formal probe.

    The projection report must use the same rollout seeds and microbatches as
    the formal probe of the same state; every draw's action/shortlist/reward
    hashes are compared one by one so the cloud is provably the same
    experiment, not a re-sample.
    """
    path = proj_report or (PROBE_ROOT / PROJ_STEM.format(state=cloud_state))
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    original = json.loads((PROBE_ROOT / RUN_STEM.format(state=cloud_state)).read_text())
    if report["rollout_seeds"] != original["rollout_seeds"]:
        raise SystemExit("Projection report uses different rollout seeds than the formal probe")

    def pick(document):
        for probe in document["probes"]:
            if probe["batches"][0]["sampler_microbatch_index"] == cloud_microbatch:
                return probe
        raise SystemExit(f"No probe at microbatch {cloud_microbatch} in {document['run']}")

    proj_probe, orig_probe = pick(report), pick(original)
    for index, (proj_pair, orig_pair) in enumerate(zip(proj_probe["draws"], orig_probe["draws"])):
        for variant in ("conditional_projection", "score_function_rloo"):
            for key in PAIR_KEYS:
                if proj_pair[variant][key] != orig_pair[variant][key]:
                    raise SystemExit(f"Draw {index} ({variant}) {key} differs from the formal "
                                     "probe; refusing to plot a different experiment")
    projections = proj_probe["projections"]
    # Recompute the per-axis variance check from the coordinates and this
    # probe's own noise variance: raw coordinates are dot products with +-1
    # sign vectors (axis norm sqrt(d)), so one axis' expected variance is the
    # full-space noise variance. Early proj reports stored noise_variance/d in
    # expected_axis_variance; the comparison is computed here either way.
    axis_variance_check = {}
    for variant, rows in projections["coordinates"].items():
        noise_variance = proj_probe["summary"][variant]["noise_variance"]
        columns = list(zip(*rows))
        means = [sum(column) / len(column) for column in columns]
        realized = [sum((value - mean) ** 2 for value in column) / (len(column) - 1)
                    for column, mean in zip(columns, means)]
        axis_variance_check[variant] = dict(
            expected=noise_variance,
            realized_over_expected_max=max(value / noise_variance for value in realized))
    return dict(path=path, state=cloud_state, microbatch=cloud_microbatch,
                seeds=report["rollout_seeds"], seed=projections["seed"],
                directions=projections["directions"],
                parameter_numel=projections["parameter_numel"],
                coords=projections["coordinates"],
                axis_variance_check=axis_variance_check)


def write_csv(rows, path):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: (f"{value:.6g}" if isinstance(value, float) else value)
                             for key, value in row.items()})


def configure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5,
        "axes.titlesize": 8.5, "axes.titlecolor": INK, "axes.titlepad": 6,
        "axes.labelsize": 8, "axes.labelcolor": INK_2,
        "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "legend.fontsize": 7,
        "text.color": INK, "figure.dpi": 300, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    return plt


def panel(axes, rows, key_cp, key_rloo, title, ylabel, reference=None, show_legend=True):
    """Two estimator clusters per fixed state; the gap is the message."""
    for group, (state, _) in enumerate(STATES):
        state_rows = sorted((row for row in rows if row["state"] == state),
                            key=lambda r: r["microbatch"])
        for column, (key, color) in enumerate(((key_cp, BLUE), (key_rloo, AQUA))):
            offsets = (-0.26, -0.18, -0.10) if column == 0 else (0.10, 0.18, 0.26)
            for offset, row in zip(offsets, state_rows):
                axes.plot(group + offset, row[key], marker="o", markersize=4.5,
                          color=color, markeredgecolor=SURFACE, markeredgewidth=1.1,
                          clip_on=False, zorder=3)
            mean = geometric_mean([row[key] for row in state_rows])
            span = (-0.32, -0.04) if column == 0 else (0.04, 0.32)
            axes.plot([group + span[0], group + span[1]], [mean, mean], color=INK,
                      linewidth=1.2, solid_capstyle="round", zorder=4)
    if reference is not None:
        axes.axhline(reference, color=MUTED, linewidth=0.8, zorder=2)
        axes.annotate("signal = noise", (-0.05, reference), xytext=(0, 3),
                      textcoords="offset points", fontsize=7, color=MUTED,
                      ha="right", va="bottom", annotation_clip=False)
    if show_legend:
        handles = [plt_line(color, label) for color, label in
                   ((BLUE, "RLOO + CMP (ours)"), (AQUA, "RLOO (no CMP)"))]
        axes.legend(handles=handles, loc="upper left", handlelength=1.2, borderaxespad=0.2)
    axes.set_yscale("log")
    axes.set_xticks(range(len(STATES)), [label for _, label in STATES])
    axes.set_xlim(-0.55, len(STATES) - 0.35)
    axes.set_title(title, loc="left")
    axes.set_ylabel(ylabel)
    axes.grid(axis="y", color=GRID, linewidth=0.6)
    axes.set_axisbelow(True)


def plt_line(color, label):
    import matplotlib.lines
    return matplotlib.lines.Line2D([], [], color=color, marker="o", markersize=4.0,
                                   markeredgecolor=SURFACE, markeredgewidth=1.0,
                                   linestyle="none", label=label)


def cloud_geometry(projection):
    """Selected-plane coordinates and derived quantities for the gradient cloud.

    x is the axis carrying the most signal (largest |CMP mean coordinate|), y the
    axis carrying the least; both are divided by the projected signal (oriented
    so the CMP cluster sits at +1), so the origin marks "no update". Shared by
    the figure and the exported figure data so the two cannot drift apart.
    """
    cp = projection["coords"]["conditional_projection"]
    rloo = projection["coords"]["score_function_rloo"]
    cp_mean = [sum(column) / len(column) for column in zip(*cp)]
    rloo_mean = [sum(column) / len(column) for column in zip(*rloo)]
    signal_axis = max(range(len(cp_mean)), key=lambda axis: abs(cp_mean[axis]))
    noise_axis = min((axis for axis in range(len(cp_mean)) if axis != signal_axis),
                     key=lambda axis: abs(cp_mean[axis]))
    sign = 1.0 if cp_mean[signal_axis] >= 0 else -1.0
    scale = abs(cp_mean[signal_axis])
    if scale <= 0:
        raise SystemExit("Projected signal is zero; cannot normalize the gradient cloud")

    def spread(values, center):
        return math.sqrt(sum((value - center) ** 2 for value in values) / (len(values) - 1))

    geometry = dict(
        signal_axis=signal_axis, noise_axis=noise_axis, sign=sign, scale=scale,
        x_cp=[sign * row[signal_axis] / scale for row in cp],
        y_cp=[row[noise_axis] / scale for row in cp],
        x_rloo=[sign * row[signal_axis] / scale for row in rloo],
        y_rloo=[row[noise_axis] / scale for row in rloo],
        cp_mean=[sign * cp_mean[signal_axis] / scale, cp_mean[noise_axis] / scale],
        rloo_mean=[sign * rloo_mean[signal_axis] / scale, rloo_mean[noise_axis] / scale],
    )
    geometry["cp_sd"] = max(spread(geometry["x_cp"], geometry["cp_mean"][0]),
                            spread(geometry["y_cp"], geometry["cp_mean"][1]))
    offset = max(abs(geometry["rloo_mean"][0] - geometry["cp_mean"][0]),
                 abs(geometry["rloo_mean"][1] - geometry["cp_mean"][1]))
    geometry["half_span"] = max(6.0 * geometry["cp_sd"], offset / 0.8)
    geometry["limit"] = 1.08 * max(abs(value) for value in
                                   geometry["x_cp"] + geometry["y_cp"]
                                   + geometry["x_rloo"] + geometry["y_rloo"])
    geometry["zoom"] = geometry["limit"] / geometry["half_span"]
    return geometry


def cloud_panel(axes, geometry, title):
    """Per-draw gradient cloud of one probe with a zoomed CMP inset."""
    x_cp, y_cp = geometry["x_cp"], geometry["y_cp"]
    x_rloo, y_rloo = geometry["x_rloo"], geometry["y_rloo"]
    limit = geometry["limit"]
    axes.axhline(0.0, color=GRID, linewidth=0.6, zorder=0)
    axes.axvline(0.0, color=GRID, linewidth=0.6, zorder=0)
    for x_left, y_left, x_right, y_right in zip(x_rloo, y_rloo, x_cp, y_cp):
        axes.plot([x_left, x_right], [y_left, y_right], color=BASELINE,
                  linewidth=0.45, alpha=0.55, zorder=1)
    axes.plot([0.0], [0.0], marker="x", color=MUTED, markersize=6.5,
              markeredgewidth=1.1, zorder=2)
    axes.annotate("origin", (0.0, 0.0), xytext=(3, -11), textcoords="offset points",
                  fontsize=7, color=MUTED)
    axes.scatter(x_rloo, y_rloo, s=13, facecolors="none", edgecolors=AQUA,
                 linewidths=1.0, zorder=3, label="RLOO (no CMP)")
    axes.scatter(x_cp, y_cp, s=11, color=BLUE, edgecolors=SURFACE,
                 linewidths=0.6, zorder=4, label="RLOO + CMP (ours)")
    axes.set_xlim(-limit, limit)
    axes.set_ylim(-limit, limit)
    axes.set_aspect("equal")
    axes.set_title(title, loc="left")
    axes.set_xlabel("signal-axis component / projected signal")
    axes.set_ylabel("noise-axis component / projected signal")
    axes.legend(loc="upper left", handlelength=1.2, borderaxespad=0.2)

    # The CMP cluster is ~20x smaller than the RLOO cloud and collapses to a
    # dot at the cloud's scale; repeat it in a zoom inset at its own size, with
    # both estimators' mean markers -- at this scale the RLOO mean lands inside
    # the CMP cluster, which is the visual unbiasedness cue.
    # Lower left: the RLOO cloud hugs the positive-signal side, so this corner
    # covers the fewest points (checked against the plotted coordinates).
    cp_x, cp_y = geometry["cp_mean"]
    rl_x, rl_y = geometry["rloo_mean"]
    half_span = geometry["half_span"]
    inset = axes.inset_axes([0.055, 0.07, 0.395, 0.395])
    inset.set_facecolor(SURFACE)
    inset.set_xlim(cp_x - half_span, cp_x + half_span)
    inset.set_ylim(cp_y - half_span, cp_y + half_span)
    inset.set_aspect("equal")
    inset.scatter(x_cp, y_cp, s=8, color=BLUE, edgecolors=SURFACE,
                  linewidths=0.5, zorder=4)
    inset.plot([cp_x], [cp_y], marker="+", color=INK, markersize=9,
               markeredgewidth=1.1, zorder=5)
    inset.plot([rl_x], [rl_y], marker="+", color=INK_2, markersize=9,
               markeredgewidth=1.1, zorder=5)
    inset.annotate("CMP mean", (cp_x, cp_y), xytext=(4, -10), textcoords="offset points",
                   fontsize=6, color=INK, zorder=6,
                   bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.8, alpha=0.85))
    inset.annotate("RLOO mean", (rl_x, rl_y), xytext=(4, 4), textcoords="offset points",
                   fontsize=6, color=INK_2, zorder=6,
                   bbox=dict(facecolor=SURFACE, edgecolor="none", pad=0.8, alpha=0.85))
    inset.tick_params(labelleft=False, labelbottom=False, length=0)
    for spine in inset.spines.values():
        spine.set_edgecolor(BASELINE)
        spine.set_linewidth(0.8)
    inset.set_title(f"CMP (×{geometry['zoom']:.0f} zoom)", fontsize=6.5, color=INK, pad=2.5)
    axes.indicate_inset_zoom(inset, edgecolor=MUTED, linewidth=0.8)


def write_figure_data(projection, geometry, rows, output_dir):
    """The compact file set every figure point can be recomputed from."""
    stem = f"gradient_cloud_{projection['state']}_micro{projection['microbatch']}"
    with (output_dir / f"{stem}.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["draw", "estimator", "x_signal_axis", "y_noise_axis",
                         "signal_axis_dot", "noise_axis_dot"])
        for variant, label, xs, ys in (
                ("score_function_rloo", "rloo", geometry["x_rloo"], geometry["y_rloo"]),
                ("conditional_projection", "cmp", geometry["x_cp"], geometry["y_cp"])):
            for draw, (x, y, row) in enumerate(zip(xs, ys, projection["coords"][variant])):
                writer.writerow([draw, label, f"{x:.6g}", f"{y:.6g}",
                                 f"{row[geometry['signal_axis']]:.6g}",
                                 f"{row[geometry['noise_axis']]:.6g}"])
    meta = dict(
        panel="gradient_probe_k0_pairwise panel (a)",
        state=projection["state"], microbatch=projection["microbatch"],
        draws=len(projection["seeds"]),
        projection=dict(
            seed=projection["seed"], directions=projection["directions"],
            parameter_numel=projection["parameter_numel"],
            axes="fixed seed-keyed Rademacher +-1 sign vectors over all trainable coordinates"),
        plane=dict(
            signal_axis=geometry["signal_axis"], noise_axis=geometry["noise_axis"],
            sign=geometry["sign"], projected_signal_scale=geometry["scale"],
            selection="signal axis = largest |CMP mean coordinate|, noise axis = smallest; "
                      "x = sign * signal_axis_dot / projected_signal_scale orients the "
                      "CMP cluster at +1"),
        cmp_mean=geometry["cp_mean"], rloo_mean=geometry["rloo_mean"],
        cmp_sd=geometry["cp_sd"], inset_half_span=geometry["half_span"],
        inset_view_zoom=geometry["zoom"],
        axis_variance_check=projection["axis_variance_check"],
        pairing="same draw index = same rollout seed; per-draw hashes verified identical "
                "across estimators and against the formal probe",
        source=projection["path"].name,
    )
    (output_dir / f"{stem}_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    with (output_dir / "snr_probes.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["state", "microbatch", "snr_cp", "snr_rloo"])
        for row in rows:
            writer.writerow([row["state"], row["microbatch"],
                             f"{row['snr_cp']:.6g}", f"{row['snr_rloo']:.6g}"])


def write_figure(rows, path, projection, geometry):
    plt = configure_matplotlib()
    if projection is None:
        # No projection readout yet: keep the previous noise + SNR design.
        print("Projection readout missing; writing the noise + SNR panels")
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(5.2, 2.6))
        panel(ax_a, rows, "noise_rms_cp", "noise_rms_rloo",
              "(a) Per-step gradient noise", "Gradient noise (RMS)")
        panel(ax_b, rows, "snr_cp", "snr_rloo",
              "(b) Signal-to-noise per step", "Signal-to-noise ratio", reference=1.0,
              show_legend=False)
        for axes in (ax_a, ax_b):
            axes.set_xlabel("Fixed state")
        ax_a.set_ylim(0.7, 90)
        ax_b.set_ylim(0.12, 12)
        fig.subplots_adjust(left=0.115, right=0.97, bottom=0.16, top=0.86, wspace=0.5)
    else:
        state_label = dict(STATES)[projection["state"]]
        fig, (ax_a, ax_b) = plt.subplots(
            1, 2, figsize=(5.4, 2.75), gridspec_kw=dict(width_ratios=(1.08, 1.0)))
        cloud_panel(ax_a, geometry, f"(a) {len(projection['seeds'])} rollout gradients, {state_label}")
        panel(ax_b, rows, "snr_cp", "snr_rloo",
              "(b) Signal-to-noise per step", "Signal-to-noise ratio", reference=1.0,
              show_legend=False)
        ax_b.set_xlabel("Fixed state")
        fig.subplots_adjust(left=0.165, right=0.965, bottom=0.225, top=0.86, wspace=0.62)
    for suffix in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{suffix}"))
    plt.close(fig)


def write_readme(rows, path, git, projection):
    unresolved = sum(1 for row in rows if not row["mean_difference_resolved"])
    worst_bias = max(row["bias_fraction_of_signal"] for row in rows)
    worst_resid = max(row["cp_residual_relative_max"] for row in rows
                      if row["cp_residual_relative_max"] is not None)
    reward_gaps = {row["pairwise_reward_agreement_gap"] for row in rows}
    state_lines = []
    for state, label in STATES:
        state_rows = [row for row in rows if row["state"] == state]
        ratio = geometric_mean([row["variance_ratio"] for row in state_rows])
        vt = geometric_mean([row["variance_time_ratio"] for row in state_rows])
        signal_cp = statistics.mean(row["signal_squared_cp"] for row in state_rows)
        signal_rloo = statistics.mean(row["signal_squared_rloo"] for row in state_rows)
        state_lines.append(
            f"| {label} | {ratio:.4f} | {vt:.4f} | {1/ratio:.0f}× | {signal_cp:.1f} / {signal_rloo:.1f} |")
    time_note = ("耗时与比值不单独成面板：CMP 每趟前后向仅慢 ~13%，`V_CMP/V_RLOO` 0.002–0.003"
                 "（降噪 300–600 倍，等效于把 ~300–600 个独立 rollout 梯度平均才追上 CMP 单步），"
                 "`V×t` 比值 0.002–0.004，见 `probes.csv` 的 `variance_ratio`/`variance_time_ratio` 列。")
    if projection is not None:
        check = projection["axis_variance_check"]
        axis_check = " / ".join(
            f"{label} {check[variant]['realized_over_expected_max']:.2f}×"
            for variant, label in (("conditional_projection", "CMP"),
                                   ("score_function_rloo", "RLOO")))
        figure_lines = (
            f"- `gradient_probe_k0_pairwise.pdf` / `.png` — 正文 F1。面板 A：**单 draw 梯度云**"
            f"（{dict(STATES)[projection['state']]}、microbatch 位置 {projection['microbatch']} 的同一批 "
            f"{len(projection['seeds'])} 次 rollout：完整参数梯度在 {projection['directions']} 条固定随机 ±1 轴"
            "上的坐标；横轴取 CMP 均值分量最大的轴、纵轴取分量最小者，均按投影信号归一；灰线连接同一 draw "
            "的两个估计器）。RLOO 的云把原点罩在里面（单步方向被噪声淹没），CMP 收拢成与原点分离的小团；"
            "右上插图按 CMP 自身尺度放大该团（64 个 draw 加两估计器均值，RLOO 均值落在 CMP 团簇内部——"
            "无偏的视觉版）。面板 B：**单步信噪比**（`√signal²/noise_rms`，对数轴，"
            "\"signal = noise\" 参考线）：CMP 4–6.5，RLOO 0.23–0.32。" + time_note)
        protocol_extra = (
            f"\n\n梯度云面板的数据来自同协议的投影补采（`{PROJ_STEM.format(state=projection['state'])}`，"
            f"不入库）：同种子、同 microbatch、关分项 pass，逐 draw 记录完整梯度在 "
            f"{projection['directions']} 条固定 ±1 随机轴（种子 {projection['seed']}，两估计器与各 probe "
            "共用）上的点积坐标；汇总脚本逐 draw 校验动作/shortlist/graded reward/pairwise 输入 hash 与正式 "
            f"probe 完全一致后才绘图。坐标为与范数 √d 的符号向量的点积，故单轴方差期望即全空间 "
            f"`noise_variance`；各轴最大实测/期望比：{axis_check}"
            f"（n={len(projection['seeds'])} 的 χ² 波动内）。2D 投影按期望保持相对散布，定量结论以面板 B 与 "
            "`probes.csv` 为准。")
    else:
        figure_lines = (
            "- `gradient_probe_k0_pairwise.pdf` / `.png` — 正文 F1。面板 A：**每步梯度噪声**"
            "（同一固定批次 64 次 rollout 的完整参数梯度围绕均值的标准差，RMS，对数轴）；"
            "点 = 单个 probe（一个固定训练 microbatch），短横线 = 三 microbatch 几何均值。"
            "面板 B：**单步信噪比**（`√signal²/noise_rms`，对数轴，\"signal = noise\" 参考线）："
            "无 CMP 时单步噪声约为信号本身的 3–4 倍（SNR<1），加 CMP 后信号是噪声的 4–6 倍。" + time_note)
        protocol_extra = ""
    cloud_files = (
        f"- `gradient_cloud_{projection['state']}_micro{projection['microbatch']}.csv` + "
        "`_meta.json` — 面板 A 的逐 draw 数据：64 draw × 两估计器的归一化坐标与所选两轴的原始点积；"
        "meta 记录轴选择/定向、投影信号尺度、两估计器均值、CMP σ、插图窗口与缩放、轴方差校验、"
        "配对与 hash 校验说明（同 draw 序号 = 同一 rollout）。\n"
        "- `snr_probes.csv` — 面板 B 的 9 个点（state / microbatch / snr_cp / snr_rloo）。\n"
        if projection is not None else "")
    readme = f"""# F1：K=0 + pairwise 主配方固定状态梯度探针

[分析计划 §F1](../../ANALYSIS_PLAN.md) 的交付目录。数据源：`outputs/f1_k0_probe/pairwise050_seed3407_{{e0,step50,step113}}_micro0-100-200.json`（探针由 `scripts/diagnose_rollout_gradients.py --compare-gradient-estimators` 生成，本目录由 `scripts/analyze_f1_probe.py` 汇总，{time.strftime('%Y-%m-%d %H:%M')}，git `{git['commit'][:8]}`{' (dirty)' if git['dirty'] else ''}）。

## 图

{figure_lines}

## 结果

| 固定状态 | V_CMP/V_RLOO（几何均值） | (V·t)_CMP/(V·t)_RLOO | 降噪倍数 | signal² CMP / RLOO |
|---|---:|---:|---:|---:|
{chr(10).join(state_lines)}

- **CMP 把完整 encoder 梯度的 rollout 方差降低约 300–600 倍**，三个固定状态、九个 probe 一致；扣除前后向耗时后（CMP 慢 ~13%）仍低约两个半数量级。
- 无偏性：{unresolved}/9 个 probe 的两估计器均值梯度差**未分辨**（无偏平方差 ≤ 0）；其余分辨出的偏差 ≤ signal² 的 {worst_bias:.0%}。两估计器的 signal² 估计逐 probe 相符。
- 配对完整性：所有 probe 的 pairwise reward 逐对差 = {sorted(g for g in reward_gaps if g is not None)}（bit 级一致）；动作/shortlist/graded reward hash 逐 draw 相等（探针内强制校验）。
- 分项（附录素材，见 `probes.csv`）：graded 与 pairwise 梯度范数、夹角随状态变化（夹角从 E0 的 ~0.6 收缩到 step 113 的 ≈0）；分解残差 `‖g_total−(g_graded+0.5·g_pairwise)‖/‖g_total‖` 最大 {worst_resid:.1%}。

## 协议

0.6B 主配方 seed 3407（`K=0` + pairwise `λ=0.5`、RLOO+CMP、`G=64`、`ρ=0.70`、`T=1`）的固定权重：E0、step 50、step 113。每状态三个训练 microbatch（位置 0/100/200，实际 source/query ID/tensor hash 见 JSON）；每批次 64 次独立 rollout，同一 draw 的动作、reward 与 shortlist 身份在两估计器间完全配对（种子重置 + hash 校验）。RLOO 对照 = 训练 score-function 分支（graded 项）+ `rloo_pairwise_shortlist_loss`（pairwise 项，端点局部 score-function；reward/LOO/归一化与 CP 逐项一致），即完整的无 CMP 对照。关闭 dropout、无 optimizer 更新。前置校验：`--self-check` 合成张量（reward 一致、均值梯度相容、与有限差分参考一致）与 E0 32-draw precheck。{protocol_extra}

## 文件

- `probes.csv` — 每个 probe 一行：方差比、V×t、两估计器的 V/noise_rms/mean norm/signal²/耗时、均值差分辨标记与偏差占比、分项统计、来源与 hash。
{cloud_files}- 原始 JSON 不入库（`outputs/` 被忽略）；`probes.csv` 的 `git_commit` 与 `diagnostic_sha256` 可定位生成版本。

## 边界（图注须带上）

固定状态方差不等于训练 seed 方差，也不证明端到端提速或 BRIGHT 增益；方差×时间的比值假设探针 pass 的前后向成本与训练步等价（CMP 的投影计算发生在 loss 侧，训练步差异见 RL 组件表的 wall-clock）。每状态仅 3 个 probe，点不合并、不当独立 seed 用。梯度云为单一代表 probe 的 2D 随机投影，只按期望保持相对散布，不新增定量结论。
"""
    path.write_text(readme)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe-root", type=Path, default=PROBE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cloud-state", default="e0", choices=[state for state, _ in STATES],
                        help="Which fixed state's projection readout feeds the gradient cloud")
    parser.add_argument("--cloud-microbatch", type=int, default=100,
                        help="Which probe microbatch position feeds the gradient cloud")
    parser.add_argument("--proj-report", type=Path, default=None,
                        help="Projection-readout JSON; defaults to the cloud-state report")
    return parser.parse_args()


def main():
    args = parse_args()
    global PROBE_ROOT
    PROBE_ROOT = args.probe_root
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = probe_rows()
    if len(rows) != len(STATES) * 3:
        raise SystemExit(f"Expected {len(STATES) * 3} probes, found {len(rows)}")
    projection = load_projection(args.cloud_state, args.cloud_microbatch, args.proj_report)
    geometry = cloud_geometry(projection) if projection is not None else None
    write_csv(rows, args.output_dir / "probes.csv")
    write_figure(rows, args.output_dir / "gradient_probe_k0_pairwise", projection, geometry)
    if projection is not None:
        write_figure_data(projection, geometry, rows, args.output_dir)
    git = dict(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                              cwd=ROOT).strip(),
               dirty=bool(subprocess.check_output(["git", "status", "--porcelain"],
                                                  text=True, cwd=ROOT).strip()))
    write_readme(rows, args.output_dir / "README.md", git, projection)
    source = (f"cloud from {projection['path'].name}, microbatch {projection['microbatch']}"
              if projection else "no projection readout; noise + SNR panels")
    print(f"Wrote {len(rows)} probe rows, figure and README to {args.output_dir} ({source})")


if __name__ == "__main__":
    main()
