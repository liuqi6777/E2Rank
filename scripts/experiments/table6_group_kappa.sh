#!/bin/bash
# Tables 6-7 -- group size / rollout structure (G1-G7) and exploration scale (K1-K6).
# Single seed by default; G=32 product = C7 (reuse).
SEEDS="${SEEDS:-42}"
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
prepare_stage1

for seed in $SEEDS; do
  for g in 4 8 16 64; do
    train_rl "g-product-${g}" "$seed" configs/grpo/grid.yaml configs/reward/ndcg.yaml stage1 --group_size "$g"
  done
  for g in 32 64; do
    train_rl "g-diagonal-${g}" "$seed" configs/grpo/abl_diagonal.yaml configs/reward/ndcg.yaml stage1 --group_size "$g"
  done

  for k in 200 400 1500 4000; do
    train_rl "k-${k}" "$seed" configs/grpo/grid.yaml configs/reward/ndcg.yaml stage1 --kappa "$k"
  done
  train_rl "k-learnable" "$seed" configs/grpo/abl_sigma_learnable.yaml configs/reward/ndcg.yaml stage1
done
