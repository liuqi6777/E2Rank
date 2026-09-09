#!/bin/bash
# S1-S3: 200-step interface and diagnostics checks. These are never paper rows.

POSTTRAIN_TRAIN_CONFIG="${POSTTRAIN_TRAIN_CONFIG:-configs/train/posttrain_smoke.yaml}"
source "$(dirname "${BASH_SOURCE[0]}")/_posttrain_common.sh"

train_supervised_posttrain "s1-cl"      configs/baseline/default.yaml
train_supervised_posttrain "s2-ranknet" configs/baseline/ranknet.yaml
train_rl_posttrain         "s3-rl-ndcg" "$POSTTRAIN_GRPO_CONFIG" configs/reward/ndcg_listwise.yaml
