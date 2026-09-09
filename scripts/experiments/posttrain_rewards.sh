#!/bin/bash
# R1-R6: reward programmability from the same public embedding checkpoint.
# R1 is C3 and is reused rather than retrained.
# R5/R6 are own-slate permutation-native rewards because cross-query documents
# have no teacher order and must not be assigned an arbitrary position.

source "$(dirname "${BASH_SOURCE[0]}")/_posttrain_common.sh"

train_rl_posttrain "c3-rl-ndcg-ib"    "$POSTTRAIN_GRPO_CONFIG" configs/reward/ndcg_listwise_in_batch.yaml
train_rl_posttrain "r2-rl-mrr-ib"     "$POSTTRAIN_GRPO_CONFIG" configs/reward/mrr_listwise.yaml
train_rl_posttrain "r3-rl-infonce-ib" "$POSTTRAIN_GRPO_CONFIG" configs/reward/infonce_listwise.yaml
train_rl_posttrain "r5-rl-top-weighted-pairwise" "$POSTTRAIN_GRPO_CONFIG" configs/reward/top_weighted_pairwise_listwise.yaml
train_rl_posttrain "r6-rl-rbo"                   "$POSTTRAIN_GRPO_CONFIG" configs/reward/rbo_listwise.yaml

if [ "${RUN_OPTIONAL_MIXTURE:-0}" = "1" ]; then
  train_rl_posttrain "r4-rl-mixture-ib" "$POSTTRAIN_GRPO_CONFIG" configs/reward/mixture_listwise.yaml
else
  echo "[R4] optional mixture skipped; set RUN_OPTIONAL_MIXTURE=1 to run it."
fi
