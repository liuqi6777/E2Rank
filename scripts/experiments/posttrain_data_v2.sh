#!/bin/bash
# D1-D2: compare continued CL with RL on the augmented train_v2 corpus.
# Both rows share the same initialization, data, budget, and candidate pool.

V2_DATASET_CONFIG="${V2_DATASET_CONFIG:-configs/dataset/e2rank_listwise_v2.yaml}"
POSTTRAIN_DATASET="$V2_DATASET_CONFIG"
source "$(dirname "${BASH_SOURCE[0]}")/_posttrain_common.sh"

train_supervised_posttrain "d1-v2-clcl-ib" \
  configs/baseline/infonce_in_batch.yaml
train_rl_posttrain "d2-v2-rl-ndcg-ib" \
  "$POSTTRAIN_GRPO_CONFIG" configs/reward/ndcg_listwise_in_batch.yaml
