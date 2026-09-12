#!/usr/bin/env bash
# Binary nDCG sweep; reuse the completed G1-A-Binary result at kappa=755.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

action="${1:-check}"
gpus="${2:-8}"
if [[ "$action" != check && "$action" != train ]]; then
  echo "Usage: bash scripts/run_g1_binary_ndcg_exploration.sh [check|train] [gpus]" >&2
  exit 2
fi
if [[ ! "$gpus" =~ ^[1-9][0-9]*$ ]]; then
  echo "gpus must be a positive integer" >&2
  exit 2
fi

runs=(
  G1-A-BinaryNDCGAlign040
  G1-A-BinaryNDCGAlign065
  G1-A-BinaryNDCGAlign080
  G1-A-BinaryNDCGAlign090
  G1-A-BinaryNDCGAlign095
)

for run in "${runs[@]}"; do
  python scripts/experiment.py check "$run" --gpus "$gpus"
done
if [[ "$action" == train ]]; then
  for run in "${runs[@]}"; do
    python scripts/experiment.py train "$run" --gpus "$gpus"
  done
fi
