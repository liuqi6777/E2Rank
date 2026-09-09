#!/bin/bash
# C0-C3: the decisive post-training comparison. Every trained row starts from
# the same public embedding checkpoint and sees the same listwise data/budget.

source "$(dirname "${BASH_SOURCE[0]}")/_posttrain_common.sh"

echo "[C0] initialization, evaluation only: $INIT_MODEL_ID"
if [ "${EVAL_INITIALIZATION:-0}" = "1" ]; then
  evaluate_model "$INIT_MODEL_ID" "MTEB(eng, v2)"
fi

train_supervised_posttrain "c1-clcl-ib"    configs/baseline/infonce_in_batch.yaml
train_supervised_posttrain "c2-ranknet-ib" configs/baseline/ranknet_in_batch.yaml
train_rl_posttrain         "c3-rl-ndcg-ib" "$POSTTRAIN_GRPO_CONFIG" configs/reward/ndcg_listwise_in_batch.yaml
