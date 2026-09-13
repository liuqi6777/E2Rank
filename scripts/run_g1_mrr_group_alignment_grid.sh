#!/usr/bin/env bash
# Six missing cells in the G={16,32,64} x alignment={0.80,0.90,0.95} grid.
# The three G=32 cells reuse completed runs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

action="${1:-check}"
gpus="${2:-8}"
if [[ "$action" != check && "$action" != train ]]; then
  echo "Usage: bash scripts/run_g1_mrr_group_alignment_grid.sh [check|train] [gpus]" >&2
  exit 2
fi
if [[ ! "$gpus" =~ ^[1-9][0-9]*$ ]]; then
  echo "gpus must be a positive integer" >&2
  exit 2
fi

# Run the alignment=0.90 anchors first so an interrupted batch still measures
# the main group-size effect around the selected recipe.
runs=(
  G1-A-MRRG16Align090
  G1-A-MRRG64Align090
  G1-A-MRRG16Align080
  G1-A-MRRG16Align095
  G1-A-MRRG64Align080
  G1-A-MRRG64Align095
)

for run in "${runs[@]}"; do
  python scripts/experiment.py check "$run" --gpus "$gpus"
done
if [[ "$action" == train ]]; then
  for run in "${runs[@]}"; do
    python scripts/experiment.py train "$run" --gpus "$gpus"
  done
fi
