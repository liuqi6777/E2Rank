#!/bin/bash

set -e

export FORCE_TORCHRUN=1
export NNODES=1
export NODE_RANK=8
export WANDB_PROJECT=${WANDB_PROJECT:-E2Rank-RL}


model_name_or_path=Alibaba-NLP/E2Rank-0.6B-Embedding-Only
data_path=data/train.jsonl
model_name=E2Rank-Full-GRPO-0.6B
query_reward_type=${QUERY_REWARD_TYPE:-ndcg}
listwise_reward_type=${LISTWISE_REWARD_TYPE:-ndcg}
query_reward_ndcg_k=${QUERY_REWARD_NDCG_K:-10}
listwise_reward_ndcg_k=${LISTWISE_REWARD_NDCG_K:-16}
query_mixed_contrastive_weight=${QUERY_MIXED_CONTRASTIVE_WEIGHT:-1.0}
query_mixed_ndcg_weight=${QUERY_MIXED_NDCG_WEIGHT:-1.0}
listwise_mixed_contrastive_weight=${LISTWISE_MIXED_CONTRASTIVE_WEIGHT:-1.0}
listwise_mixed_ndcg_weight=${LISTWISE_MIXED_NDCG_WEIGHT:-1.0}
query_contrastive_use_in_batch_negatives=${QUERY_CONTRASTIVE_USE_IN_BATCH_NEGATIVES:-False}
listwise_contrastive_use_in_batch_negatives=${LISTWISE_CONTRASTIVE_USE_IN_BATCH_NEGATIVES:-False}


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
  --query_reward_type $query_reward_type \
  --listwise_reward_type $listwise_reward_type \
  --query_reward_ndcg_k $query_reward_ndcg_k \
  --listwise_reward_ndcg_k $listwise_reward_ndcg_k \
  --query_mixed_contrastive_weight $query_mixed_contrastive_weight \
  --query_mixed_ndcg_weight $query_mixed_ndcg_weight \
  --listwise_mixed_contrastive_weight $listwise_mixed_contrastive_weight \
  --listwise_mixed_ndcg_weight $listwise_mixed_ndcg_weight \
  --query_contrastive_use_in_batch_negatives $query_contrastive_use_in_batch_negatives \
  --listwise_contrastive_use_in_batch_negatives $listwise_contrastive_use_in_batch_negatives \
  --listwise_loss_weight 1.0 \
  --advantage_norm True \
  --query_relevance_scheme binary \
  --listwise_relevance_scheme graded
