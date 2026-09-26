# F3: paired-seed BRIGHT domain differences

Source: `paper/_summary/k0_bright/subset_summary.csv` (original-query 0.6B RELER K=0, pairwise weight 0.5, versus InfoNCE). Three paired training seeds 42/3407/2026; nDCG@10 values are multiplied by 100. `paired_subset_deltas.csv` contains all plotted per-seed points and three-seed means. Recreate with `python scripts/plot_f3_domain_deltas.py`.

The equal-subset macro difference from these four-decimal source scores is 0.735 points. Subsets are evaluation domains, not independent training repeats.
