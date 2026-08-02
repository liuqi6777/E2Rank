#!/bin/bash
# Table 3 (tab:ablation-components) -- runs A1-A5. A1 = C3, A4 = C7 (reuse).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
prepare_stage1

for seed in $SEEDS; do
  train_rl "a2-query-only"  "$seed" configs/grpo/default.yaml        configs/reward/ndcg.yaml stage1
  train_rl "a3-docs-only"   "$seed" configs/grpo/documents_only.yaml configs/reward/ndcg.yaml stage1
  # A5 carries a G^3 reward tensor. If it OOMs, drop BOTH A5 and A4 to G=16 so the
  # comparison stays matched, and say so in the caption.
  train_rl "a5-factorized"  "$seed" configs/grpo/factorized.yaml     configs/reward/ndcg.yaml stage1 \
      ${A5_GROUP_SIZE:+--group_size "$A5_GROUP_SIZE"}
done
