#!/bin/bash
# Table 4 (tab:ablation-reward) -- runs R1-R8 plus the backprop counterparts that the
# Delta column needs. R5 = C7 (reuse).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
prepare_stage1

for seed in $SEEDS; do
  # Differentiable rewards: the theory says these should land near their backprop twins.
  train_rl "r1-ctr"          "$seed" configs/grpo/grid.yaml configs/reward/contrastive_no_in_batch.yaml stage1
  train_rl "r2-ctr-ib"       "$seed" configs/grpo/grid.yaml configs/reward/contrastive_in_batch.yaml    stage1
  train_rl "r3-infonce"      "$seed" configs/grpo/grid.yaml configs/reward/infonce_no_in_batch.yaml     stage1
  train_rl "r4-infonce-ib"   "$seed" configs/grpo/grid.yaml configs/reward/infonce_in_batch.yaml        stage1

  # Exact ranking metrics: where the method is supposed to earn its keep.
  train_rl "r6-ndcg-ib"      "$seed" configs/grpo/grid.yaml configs/reward/ndcg_in_batch.yaml     stage1
  train_rl "r7-ndcg-ib-all"  "$seed" configs/grpo/grid.yaml configs/reward/ndcg_in_batch_all.yaml stage1
  train_rl "r8-mrr"          "$seed" configs/grpo/grid.yaml configs/reward/mrr.yaml               stage1

  # Backprop counterpart for the Delta column. C3 (CL->CL InfoNCE) is R4's counterpart;
  # R1-R3 need matching supervised runs over the same negative set.
  train_supervised_stage2 "r-bp-infonce" "$seed" configs/baseline/infonce.yaml
done
