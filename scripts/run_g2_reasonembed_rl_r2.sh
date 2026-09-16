#!/usr/bin/env bash
# ReasonEmbed single-seed binary CP: E -> W -> D, each followed by BRIGHT.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

action="${1:-check}"
gpus="${2:-8}"
if [[ "$action" != check && "$action" != train ]] || [[ "$gpus" != 8 ]] || (( $# > 2 )); then
  echo "Usage: bash scripts/run_g2_reasonembed_rl_r2.sh [check|train] [8]" >&2
  exit 2
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

g2rl() {
  python scripts/experiment.py "$@" --gpus "$gpus" \
    --suite configs/experiments/iclr2027/suite_g2_reasonembed_rl_r2.yaml \
    --config configs/experiments_g2_reasonembed_rl_r2.yaml
}

runs=(G2-R2-ReasonEmbed-E-RL G2-R2-ReasonEmbed-W-RL G2-R2-ReasonEmbed-D-RL)
for run in "${runs[@]}"; do
  g2rl check "$run"
done
if [[ "$action" == train ]]; then
  for run in "${runs[@]}"; do
    g2rl train "$run"
  done
fi
