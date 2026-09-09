#!/bin/bash
# Supervised (InfoNCE) launcher -- the mirror of run.sh for src/train_baseline.py.
# Drives Stage 1, the CL->CL control, and the backpropagated counterpart of the
# InfoNCE-as-reward row. Same env contract as run.sh; the only difference is the entrypoint
# and that it takes --base-baseline instead of --base-grpo / --base-reward.
# InfoNCE remains the default; RankNet is available for rank-based post-training.

set -euo pipefail

export FORCE_TORCHRUN=1
export NNODES=${NNODES:-1}
export NPROC_PER_NODE=${NPROC_PER_NODE:-${NODE_RANK:-8}}
export WANDB_PROJECT=${WANDB_PROJECT:-E2Rank-RL}

if [ $# -gt 0 ]; then
  first_arg=$1
  if [[ "$first_arg" == *.yaml || "$first_arg" == *.yml || "$first_arg" == *.json ]]; then
    if [ ! -f "$first_arg" ]; then
      echo "Config file not found: $first_arg" >&2
      exit 1
    fi
  fi
fi

torchrun \
  --nnodes="$NNODES" --nproc_per_node="$NPROC_PER_NODE" \
  src/train_baseline.py \
  "$@"
