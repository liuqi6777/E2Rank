#!/bin/bash
# Phase A-C (tab:ablation-components) -- runs A2/A3/A5, plus the A-X corner check.
# A1 = C3 (phaseB_recipe.sh), A4 = C7 (the default, phaseA_mixture.sh). Both are reused.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_stage1

train_rl "a2-query-only" configs/grpo/query_only.yaml     - stage1
train_rl "a3-docs-only"  configs/grpo/documents_only.yaml - stage1

# A5 carries a G^3 reward tensor, and with an in-batch reward the last axis is B*slate wide,
# so this is the row that OOMs first. If it does, drop BOTH A5 and A4 to G=16 so the comparison
# stays matched, and say so in the caption.
train_rl "a5-factorized" configs/grpo/factorized.yaml     - stage1 \
    ${A5_GROUP_SIZE:+--group_size "$A5_GROUP_SIZE"}

# A-X: the corner. Runner-up reward at runner-up components. If it beats the default cell, the
# two factors interact, the one-factor-at-a-time design was invalid, and the sweep must widen
# to the 2x2 corners. Set both from the A-R and A-C results before running.
if [ -n "${AX_REWARD:-}" ] && [ -n "${AX_GRPO:-}" ]; then
  train_rl "ax-corner" "$AX_GRPO" "$AX_REWARD" stage1
else
  echo "  [skip] A-X corner: set AX_REWARD and AX_GRPO from the A-R / A-C runner-ups"
fi
