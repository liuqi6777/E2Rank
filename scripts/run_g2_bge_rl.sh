#!/usr/bin/env bash
# BGE-M3 data, Qwen initializations: original E0, same-data CL warm-up W0, then B0.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

action="${1:-check}"
gpus="${2:-8}"
if [[ "$action" != check && "$action" != train ]]; then
  echo "Usage: bash scripts/run_g2_bge_rl.sh [check|train] [gpus]" >&2
  exit 2
fi
if [[ ! "$gpus" =~ ^[1-9][0-9]*$ ]]; then
  echo "gpus must be a positive integer" >&2
  exit 2
fi

runs=(G2-BGE-E-RL G2-BGE-W-RL G2-BGE-D-RL)

# W0 must be the completed BGE-D-CL output, never the E2Rank warm-up.
for run in "${runs[@]}"; do
  python scripts/experiment.py check "$run" --gpus "$gpus"
done
if [[ "$action" == train ]]; then
  for run in "${runs[@]}"; do
    python scripts/experiment.py train "$run" --gpus "$gpus"
  done
fi
