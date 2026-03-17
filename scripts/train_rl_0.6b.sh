#!/bin/bash

set -e

export FORCE_TORCHRUN=1
export NNODES=1
export NODE_RANK=8
export WANDB_PROJECT=${WANDB_PROJECT:-E2Rank-RL}


model_name_or_path=Alibaba-NLP/E2Rank-0.6B-Embedding-Only
data_path=data/train.jsonl
model_name=E2Rank-Full-GRPO-0.6B


torchrun \
  --nnodes=$NNODES --nproc_per_node=$NODE_RANK \
  src/train.py \
  --deepspeed ./scripts/zero3.json \
  --output_dir checkpoints/$model_name \
  --model_name_or_path $model_name_or_path \
  --data_path $data_path \
  --bf16 \
  --tf32 True \
  --per_device_train_batch_size 4 \
  --gradient_checkpointing \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-6 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "linear" \
  --num_train_epochs 1 \
  --logging_strategy "steps" \
  --logging_steps 1 \
  --report_to "wandb" \
  --run_name $model_name \
  --save_strategy "steps" \
  --save_steps 200 \
  --overwrite_output_dir \
  --lora_enabled True \
  --lora_r 64 \
  --rl_mode dual \
  --group_size 8 \
  --sigma 0.05 \
  --query_reward_ndcg_k 10 \
  --listwise_reward_ndcg_k 16 \
  --listwise_loss_weight 1.0 \
  --advantage_norm True \
  --query_relevance_scheme binary \
  --listwise_relevance_scheme graded
