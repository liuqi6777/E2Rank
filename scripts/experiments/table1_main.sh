#!/bin/bash
# Table 1 (MTEB eng v2) -- runs M1-M3: the CL->RL recipe and its CL-only reference
# at 0.6B / 4B / 8B.  0.6B reuses C2/C7 from Table 2, so run table2_recipe.sh first.
#
#   SCALE=4b SEEDS=42 bash scripts/experiments/table1_main.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

prepare_stage1

for seed in $SEEDS; do
  # CL-only reference row (Stage 1 alone, already trained by prepare_stage1).
  echo "  [note] CL-only row for seed $seed is $(stage1_dir "$seed")"
  # CL->RL: the model reported in the table.
  train_rl "m-clrl" "$seed" configs/grpo/grid.yaml configs/reward/ndcg.yaml stage1
done

echo
echo "Evaluate with:  bash scripts/experiments/eval_all.sh full"
