#!/bin/bash
# Table 5 (tab:ablation-estimator) -- runs E1-E9. E1/E3/E6/E8 = C7 or R6 (reuse).
# The frozen-candidate rows need an in-batch reward: the rescale is a no-op without one.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
prepare_stage1

for seed in $SEEDS; do
  # E2  projected Gaussian at matched mean alignment (A_d(755)=0.530=1/sqrt(1+0.05^2*1024)).
  train_rl "e2-gaussian"      "$seed" configs/grpo/abl_gaussian.yaml   configs/reward/ndcg.yaml stage1
  # E4/E5  advantage normalization.
  train_rl "e4-adv-shared"    "$seed" configs/grpo/abl_adv_shared.yaml configs/reward/ndcg.yaml stage1
  train_rl "e5-adv-none"      "$seed" configs/grpo/abl_adv_none.yaml   configs/reward/ndcg.yaml stage1
  # E7  PREDICTED FAILURE: reward/std -> 0 and degenerate_frac -> 1. Report the diagnostics
  #     even though the run is useless downstream, and repeat at kappa=400 where 1/A_d is ~2.9.
  train_rl "e7-no-rescale"    "$seed" configs/grpo/abl_no_frozen_rescale.yaml configs/reward/ndcg_in_batch.yaml stage1
  train_rl "e7-no-rescale-k400" "$seed" configs/grpo/abl_no_frozen_rescale.yaml configs/reward/ndcg_in_batch.yaml stage1 --kappa 400
  # E9  in-batch candidates scored with sampled embeddings (cross-sample leakage).
  train_rl "e9-ib-sampled"    "$seed" configs/grpo/abl_in_batch_sampled.yaml configs/reward/ndcg_in_batch.yaml stage1
done
