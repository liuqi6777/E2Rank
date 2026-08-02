#!/bin/bash
# Appendix Table X -- secondary single-flag ablations (X1-X7). X6 = k-learnable (reuse).
SEEDS="${SEEDS:-42}"
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
prepare_stage1

for seed in $SEEDS; do
  for k in 4 10; do
    train_rl "x-ndcgk-${k}" "$seed" configs/grpo/grid.yaml configs/reward/ndcg.yaml stage1 --reward_ndcg_k "$k"
  done
  # X3 binary labels: swaps the dataset slot, so it bypasses the train_rl helper.
  out="$(run_dir "x3-binary" "$seed")"
  already_done "$out" || launch bash ./scripts/run.sh \
    --base-train configs/train/stage2.yaml --base-dataset configs/dataset/stage2_binary.yaml \
    --base-model "$BASE_MODEL_CONFIG" --model_name_or_path "$(stage1_merged_dir "$seed")" \
    --base-grpo configs/grpo/grid.yaml --base-reward configs/reward/ndcg.yaml \
    --base-eval configs/eval/default.yaml \
    --seed "$seed" --run_name "$(basename "$out")" --output_dir "$out"

  for b in 0.001 0.01; do
    train_rl "x-kl-${b}" "$seed" configs/grpo/abl_kl.yaml configs/reward/ndcg.yaml stage1 --kl_coef "$b"
  done

  # X7 two epochs: swaps the train slot.
  out="$(run_dir "x7-2ep" "$seed")"
  already_done "$out" || launch bash ./scripts/run.sh \
    --base-train configs/train/stage2_2ep.yaml --base-dataset configs/dataset/stage2.yaml \
    --base-model "$BASE_MODEL_CONFIG" --model_name_or_path "$(stage1_merged_dir "$seed")" \
    --base-grpo configs/grpo/grid.yaml --base-reward configs/reward/ndcg.yaml \
    --base-eval configs/eval/default.yaml \
    --seed "$seed" --run_name "$(basename "$out")" --output_dir "$out"

  # Full fine-tuning (the appendix adaptation row). No KL anchor available here.
  out="$(run_dir "x-full-ft" "$seed")"
  already_done "$out" || launch bash ./scripts/run.sh \
    --base-train configs/train/stage2_full_ft.yaml --base-dataset configs/dataset/stage2.yaml \
    --base-model "$BASE_MODEL_CONFIG" --model_name_or_path "$(stage1_merged_dir "$seed")" \
    --base-grpo configs/grpo/grid.yaml --base-reward configs/reward/ndcg.yaml \
    --base-eval configs/eval/default.yaml \
    --seed "$seed" --run_name "$(basename "$out")" --output_dir "$out"
done
