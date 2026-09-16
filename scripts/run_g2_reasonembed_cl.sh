#!/usr/bin/env bash
# ReasonEmbed G2 CL queue: D -> E -> W, with final BRIGHT after each run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

action="${1:-check}"
gpus="${2:-8}"
if [[ "$action" != check && "$action" != train ]] || [[ "$gpus" != 8 ]] || (( $# > 2 )); then
  echo "Usage: bash scripts/run_g2_reasonembed_cl.sh [check|train] [8]" >&2
  exit 2
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

g2cl() {
  python scripts/experiment.py "$@" --gpus "$gpus" \
    --suite configs/experiments/iclr2027/suite_g2_reasonembed_cl.yaml \
    --config configs/experiments_g2_reasonembed_cl.yaml
}

# W's model prerequisite exists only after D has finished.
for run in G2-ReasonEmbed-D-CL G2-ReasonEmbed-E-CL; do
  g2cl check "$run"
done
if [[ "$action" == check ]]; then
  g2cl show G2-ReasonEmbed-W-CL
  echo "D/E preflight passed; W's model dependency will be checked after D completes."
else
  for run in G2-ReasonEmbed-D-CL G2-ReasonEmbed-E-CL G2-ReasonEmbed-W-CL; do
    g2cl check "$run"
    g2cl train "$run"
  done
fi
