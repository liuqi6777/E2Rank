#!/usr/bin/env bash
# Seven independent E0 ablations; reuse the completed MRRAlign090 control.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

action="${1:-check}"
gpus="${2:-8}"
if [[ "$action" != check && "$action" != train ]]; then
  echo "Usage: bash scripts/run_g1_mrr090_ablations.sh [check|train] [gpus]" >&2
  exit 2
fi
if [[ ! "$gpus" =~ ^[1-9][0-9]*$ ]]; then
  echo "gpus must be a positive integer" >&2
  exit 2
fi

runs=(
  G1-A-MRR090-QPolicy
  G1-A-MRR090-DPolicy
  G1-A-MRR090-Paired
  G1-A-MRR090-Cal
  G1-A-MRR090-Norm
  G1-A-MRR090-DocMean
  G1-A-MRR090-NormDocMean
)

# Preflight every run before training. Stop on failure; never overwrite or retry.
for run in "${runs[@]}"; do
  python scripts/experiment.py check "$run" --gpus "$gpus"
done
if [[ "$action" == train ]]; then
  for run in "${runs[@]}"; do
    python scripts/experiment.py train "$run" --gpus "$gpus"
  done
fi
