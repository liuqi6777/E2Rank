#!/bin/bash
# Post-hoc MTEB evaluation for post-training checkpoints.
# Usage: bash scripts/experiments/posttrain_eval.sh [full|subset]

source "$(dirname "${BASH_SOURCE[0]}")/_posttrain_common.sh"

MODE="${1:-full}"
case "$MODE" in
  full) BENCHMARK="MTEB(eng, v2)" ;;
  subset) BENCHMARK="MTEB(eng, v1, subset)" ;;
  *) echo "Usage: $0 [full|subset]" >&2; exit 1 ;;
esac

CKPT_GLOB="${CKPT_GLOB:-${CKPT_ROOT}/*}"
if [ "${EVAL_INITIALIZATION:-1}" = "1" ]; then
  evaluate_model "$INIT_MODEL_ID" "$BENCHMARK" "$INIT_MODEL_CONFIG"
fi

for checkpoint in $CKPT_GLOB; do
  [ -d "$checkpoint" ] || continue
  evaluate_model "$checkpoint" "$BENCHMARK"
done
