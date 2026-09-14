#!/usr/bin/env bash
# Repeat the selected MRR@10 / G=32 / alignment=0.90 recipe from E0.
# Reuse prepared data and the completed seed-42 result; do not retrain seed 42.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

action="${1:-check}"
gpus="${2:-8}"
if [[ "$action" != check && "$action" != train ]]; then
  echo "Usage: bash scripts/run_g1_mrr_seed_repeats.sh [check|train] [gpus]" >&2
  exit 2
fi
if [[ ! "$gpus" =~ ^[1-9][0-9]*$ ]]; then
  echo "gpus must be a positive integer" >&2
  exit 2
fi

runs=(
  G1-A-MRRAlign090-Seed3407
  G1-A-MRRAlign090-Seed2026
)

for run in "${runs[@]}"; do
  python scripts/experiment.py check "$run" --gpus "$gpus"
done
if [[ "$action" == train ]]; then
  for run in "${runs[@]}"; do
    python scripts/experiment.py train "$run" --gpus "$gpus"
  done
fi
