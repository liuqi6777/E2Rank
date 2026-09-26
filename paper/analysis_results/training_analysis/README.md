# Main-text training analysis

Four panels in three columns, 6.75 × 2.5 inches: wider gradient and reward panels, with narrower pairwise-weight/alignment sensitivity panels stacked on the right. The sensitivity panels use identical y-axis limits; filled blue points mark the main settings. All reward curves are solid lines.

Panel positions share explicit upper/lower boundaries. The gradient panel is wider than tall but retains equal axis scaling by extending its horizontal limits; all three bottom axis labels share a figure-level baseline.

- Gradient source: `../gradient_probe_k0_pairwise/gradient_cloud_e0_micro100.csv` and its metadata; 64 matched draws, one fixed microbatch. Axis normalization and inset limits are copied from the audited export. The cloud is illustrative; full-parameter variance ratios remain in the paper text and diagnostic table.
- Reward source: panels A and B of `../reward_curves_k0/curves.csv`; all 113 steps, three-seed means and sample SD, no smoothing. Blue = Table 2 row (a), RELER's nDCG component (`reward/mean`); green = row (c), nDCG without pairwise feedback (Graded nDCG / Product / CMP=Yes); amber = row (a), RELER's weighted pairwise component (`0.5 * reward/pairwise/mean`). The legend uses these italic row labels. The blue and amber components sum to RELER's total reward, which is not plotted. Both recipes use CMP and product rollouts. Batches change at each step, so these are training curves, not validation curves. The matched ablation establishes the +0.66 BRIGHT gain from adding pairwise feedback; similar nDCG trajectories alone do not identify its causal mechanism.
- Pairwise-weight source: `../../_summary/k0_bright/k0_bright_grouped.csv`, original-query scores for the matched Align070/K0/T1 graded-only, Pairwise050, and Pairwise100 recipes. Points show means and error bars show sample SD across three seeds. Only the weight changes; no interpolation is shown. The duplicate appendix table is no longer included.
- Alignment source: the same grouped CSV, original-query scores for Align060/070/080/090 with K0/T1/Pairwise050 fixed. Points show three-seed means and sample SD. The duplicate current-recipe alignment table is no longer included; historical local-pool controls remain separate.
- Rebuild from the repository root: `uv run --no-project --with matplotlib --with pyyaml python scripts/plot_training_analysis.py`.

Only this compact analysis figure is included in the manuscript. Older gradient, reward, difference, and case-analysis artifacts remain as source records but are no longer included in the appendix.
