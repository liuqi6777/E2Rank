#!/usr/bin/env bash
# All runs start from E0 with training/data seed 42; only action RNG changes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
action="${1:-check}"
gpus="${2:-8}"
if [[ "$action" != check && "$action" != train ]] || [[ ! "$gpus" =~ ^[1-9][0-9]*$ ]]; then
  echo "Usage: bash scripts/run_g1_rollout_seed_repeats.sh [check|train] [gpus]" >&2
  exit 2
fi
runs=(G1-A-MRR090-Rollout42 G1-A-MRR090-Rollout3407 G1-A-MRR090-Rollout2026)
for run in "${runs[@]}"; do
  python scripts/experiment.py check "$run" --gpus "$gpus"
done
if [[ "$action" == train ]]; then
  for run in "${runs[@]}"; do
    python scripts/experiment.py train "$run" --gpus "$gpus"
  done
fi
