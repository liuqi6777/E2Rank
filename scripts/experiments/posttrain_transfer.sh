#!/bin/bash
# T0-T2: optional second-initialization transfer. The caller must select and
# audit the model rather than silently treating E2Rank-on-E2Rank as independent.

source "$(dirname "${BASH_SOURCE[0]}")/_posttrain_common.sh"

: "${TRANSFER_MODEL_CONFIG:?Set TRANSFER_MODEL_CONFIG to an audited model config}"
: "${TRANSFER_MODEL_ID:?Set TRANSFER_MODEL_ID to the matching model ID/path}"

echo "[T0] transfer initialization, evaluation only: $TRANSFER_MODEL_ID"
if [ "${EVAL_INITIALIZATION:-0}" = "1" ]; then
  evaluate_model "$TRANSFER_MODEL_ID" "MTEB(eng, v2)"
fi

RUN_MODEL_CONFIG="$TRANSFER_MODEL_CONFIG" \
  train_supervised_posttrain "t1-clcl-ib" configs/baseline/infonce_in_batch.yaml
RUN_MODEL_CONFIG="$TRANSFER_MODEL_CONFIG" \
  train_rl_posttrain "t2-rl-ndcg-ib" "$POSTTRAIN_GRPO_CONFIG" configs/reward/ndcg_listwise_in_batch.yaml
