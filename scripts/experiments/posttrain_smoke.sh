#!/bin/bash
# S1-S3: 200-step interface and diagnostics checks. These are never paper rows.

POSTTRAIN_TRAIN_CONFIG="${POSTTRAIN_TRAIN_CONFIG:-configs/train/posttrain_smoke.yaml}"
source "$(dirname "${BASH_SOURCE[0]}")/_posttrain_common.sh"

train_supervised_posttrain "s1-cl-ib"      configs/baseline/infonce_in_batch.yaml
train_supervised_posttrain "s2-ranknet-ib" configs/baseline/ranknet_in_batch.yaml
train_rl_posttrain         "s3-rl-ndcg-ib" "$POSTTRAIN_GRPO_CONFIG" configs/reward/ndcg_listwise_in_batch.yaml
