#!/usr/bin/env bash
# Train only the BGE-M3 CL runs: B0, original G1 E0, then the shared B0->CL checkpoint.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

action="${1:-check}"
gpus="${2:-8}"
if [[ "$action" != check && "$action" != train ]]; then
  echo "Usage: bash scripts/run_g2_bge_cl.sh [check|train] [gpus]" >&2
  exit 2
fi
if [[ ! "$gpus" =~ ^[1-9][0-9]*$ ]]; then
  echo "gpus must be a positive integer" >&2
  exit 2
fi

# W-CL cannot pass its checkpoint check until BGE-D-CL has completed.
for run in G2-BGE-D-CL G2-BGE-E-CL; do
  python scripts/experiment.py check "$run" --gpus "$gpus"
done
if [[ "$action" == check ]]; then
  python scripts/experiment.py show G2-BGE-W-CL --gpus "$gpus"
  echo "BGE W-CL dependency check is deferred until BGE D-CL completes; use run_g2_bge_rl.sh for RL."
else
  for run in G2-BGE-D-CL G2-BGE-E-CL G2-BGE-W-CL; do
    python scripts/experiment.py check "$run" --gpus "$gpus"
    python scripts/experiment.py train "$run" --gpus "$gpus"
  done
fi
