#!/bin/bash

set -e

export FORCE_TORCHRUN=1
export NNODES=1
export NODE_RANK=8
export WANDB_PROJECT=${WANDB_PROJECT:-E2Rank-RL}

config_path=${1:-configs/exp/train_rl_0.6b.yaml}

if [ ! -f "$config_path" ]; then
  echo "Config file not found: $config_path" >&2
  exit 1
fi

torchrun \
  --nnodes=$NNODES --nproc_per_node=$NODE_RANK \
  src/train.py \
  "$config_path"
