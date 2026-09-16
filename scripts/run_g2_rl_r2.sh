#!/usr/bin/env bash
# E2Rank single-seed CP: E -> W -> D, each followed by final MTEB.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

action="${1:-check}"
gpus="${2:-8}"
if [[ "$action" != check && "$action" != train ]] || [[ "$gpus" != 8 ]] || (( $# > 2 )); then
  echo "Usage: bash scripts/run_g2_rl_r2.sh [check|train] [8]" >&2
  exit 2
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

g2rl() {
  python scripts/experiment.py "$@" --gpus "$gpus" \
    --suite configs/experiments/iclr2027/suite_g2_rl_r2.yaml \
    --config configs/experiments_g2_rl_r2.yaml
}

runs=(G2-R2-E-RL G2-R2-W-RL G2-R2-D-RL)
# W0 must already exist; this queue never schedules CL or resumes old outputs.
for run in "${runs[@]}"; do
  g2rl check "$run"
done
if [[ "$action" == train ]]; then
  for run in "${runs[@]}"; do
    g2rl train "$run"
  done
fi
