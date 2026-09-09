#!/bin/bash
# Focused appendix ablations. Usage:
#   bash scripts/experiments/posttrain_ablations.sh [all|scratch|components|estimator|rescale|group]

source "$(dirname "${BASH_SOURCE[0]}")/_posttrain_common.sh"

BLOCK="${1:-all}"
want() { [ "$BLOCK" = "all" ] || [ "$BLOCK" = "$1" ]; }

BASE_LLM_MODEL_CONFIG="${BASE_LLM_MODEL_CONFIG:-configs/model/qwen3_0.6b.yaml}"
BASE_LLM_MODEL_ID="${BASE_LLM_MODEL_ID:-Qwen/Qwen3-0.6B}"

if want scratch; then
  echo "[A0] base LLM, evaluation only: $BASE_LLM_MODEL_ID"
  if [ "${EVAL_INITIALIZATION:-0}" = "1" ]; then
    evaluate_model "$BASE_LLM_MODEL_ID" "MTEB(eng, v2)"
  fi
  RUN_MODEL_CONFIG="$BASE_LLM_MODEL_CONFIG" \
    train_supervised_posttrain "a1-base-cl" configs/baseline/default.yaml
  RUN_MODEL_CONFIG="$BASE_LLM_MODEL_CONFIG" \
    train_rl_posttrain "a2-base-rl" "$POSTTRAIN_GRPO_CONFIG" configs/reward/ndcg_listwise.yaml
fi

if want components; then
  train_rl_posttrain "a3-query-only" configs/grpo/posttrain_query_only.yaml configs/reward/ndcg_listwise.yaml
  train_rl_posttrain "a4-documents-only" configs/grpo/posttrain_documents_only.yaml configs/reward/ndcg_listwise.yaml
fi

if want estimator; then
  train_rl_posttrain "a5-gaussian" configs/grpo/posttrain_gaussian.yaml configs/reward/ndcg_listwise.yaml
fi

if want rescale; then
  # Frozen rescaling has an effect only when detached in-batch candidates are
  # present, so run an explicit matched on/off pair under the same reward.
  train_rl_posttrain "a6-rescale-control" "$POSTTRAIN_GRPO_CONFIG" \
    configs/reward/ndcg_listwise_in_batch.yaml
  train_rl_posttrain "a6-no-rescale" configs/grpo/posttrain_no_frozen_rescale.yaml \
    configs/reward/ndcg_listwise_in_batch.yaml
fi

if want group; then
  train_rl_posttrain "a7-group16" configs/grpo/posttrain_group16.yaml \
    configs/reward/ndcg_listwise.yaml
fi
