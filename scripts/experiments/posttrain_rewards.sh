#!/bin/bash
# R1-R4: reward programmability from the same public embedding checkpoint.
# R1 is C3 and is reused rather than retrained.

source "$(dirname "${BASH_SOURCE[0]}")/_posttrain_common.sh"

train_rl_posttrain "c3-rl-ndcg"    "$POSTTRAIN_GRPO_CONFIG" configs/reward/ndcg_listwise.yaml
train_rl_posttrain "r2-rl-mrr"     "$POSTTRAIN_GRPO_CONFIG" configs/reward/mrr_listwise.yaml
train_rl_posttrain "r3-rl-infonce" "$POSTTRAIN_GRPO_CONFIG" configs/reward/infonce_listwise.yaml

if [ "${RUN_OPTIONAL_MIXTURE:-0}" = "1" ]; then
  train_rl_posttrain "r4-rl-mixture" "$POSTTRAIN_GRPO_CONFIG" configs/reward/mixture_listwise.yaml
else
  echo "[R4] optional mixture skipped; set RUN_OPTIONAL_MIXTURE=1 to run it."
fi
