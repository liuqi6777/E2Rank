#!/usr/bin/env bash
# Sequential fixed-alignment MRR sweep; the completed kappa=755 point is reused.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

action="${1:-check}"
gpus="${2:-8}"
if [[ "$action" != check && "$action" != train ]]; then
  echo "Usage: bash scripts/run_g1_mrr_exploration.sh [check|train] [gpus]" >&2
  exit 2
fi
if [[ ! "$gpus" =~ ^[1-9][0-9]*$ ]]; then
  echo "gpus must be a positive integer" >&2
  exit 2
fi

runs=(
  G1-A-MRRAlign040
  G1-A-MRRAlign065
  G1-J-RL-MRRSmall
  G1-A-MRRAlign090
  G1-A-MRRAlign095
)

# Check every row before allocating training resources. No automatic overwrite/retry.
for run in "${runs[@]}"; do
  python scripts/experiment.py check "$run" --gpus "$gpus"
done
if [[ "$action" == train ]]; then
  for run in "${runs[@]}"; do
    python scripts/experiment.py train "$run" --gpus "$gpus"
  done
fi
