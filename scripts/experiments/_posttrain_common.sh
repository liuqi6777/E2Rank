#!/bin/bash
# Shared helpers for the public-checkpoint post-training experiments.
# Source this file; do not run it directly.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
CKPT_ROOT="${CKPT_ROOT:-checkpoints/posttrain}"
INIT_MODEL_CONFIG="${INIT_MODEL_CONFIG:-configs/model/qwen3_embedding_0.6b.yaml}"
INIT_MODEL_ID="${INIT_MODEL_ID:-Qwen/Qwen3-Embedding-0.6B}"
POSTTRAIN_DATASET="${POSTTRAIN_DATASET:-configs/dataset/e2rank_listwise.yaml}"
# Empty means: use the preset declared by the selected model config. Setting this
# variable remains an explicit override for calibration runs and ablations.
POSTTRAIN_TRAIN_CONFIG="${POSTTRAIN_TRAIN_CONFIG:-}"
POSTTRAIN_GRPO_CONFIG="${POSTTRAIN_GRPO_CONFIG:-configs/grpo/posttrain.yaml}"
POSTTRAIN_EVAL_CONFIG="${POSTTRAIN_EVAL_CONFIG:-configs/eval/mteb.yaml}"
export WANDB_PROJECT="${WANDB_PROJECT:-E2Rank-RL-Posttrain}"

model_tag() {
  local filename
  filename="$(basename "$1")"
  printf '%s\n' "${filename%.*}"
}

run_dir() {
  local id="$1" model_config="$2"
  printf '%s/%s-%s-s%s\n' "$CKPT_ROOT" "$id" "$(model_tag "$model_config")" "$SEED"
}

log() { printf '\n\033[1m>>> %s\033[0m\n' "$*"; }

launch() {
  log "$*"
  if [ "$DRY_RUN" = "1" ]; then return 0; fi
  "$@"
}

already_done() {
  if [ -f "$1/config.json" ] || [ -f "$1/adapter_config.json" ]; then
    echo "  [skip] $1 already contains final model artifacts"
    return 0
  fi
  return 1
}

# $1 run id, $2 baseline config, $3... command-line overrides.
# RUN_MODEL_CONFIG may override the public initialization for a single call.
train_supervised_posttrain() {
  local id="$1" baseline="$2"; shift 2
  local model_config="${RUN_MODEL_CONFIG:-$INIT_MODEL_CONFIG}"
  local out; out="$(run_dir "$id" "$model_config")"
  already_done "$out" && return 0
  local command=(bash ./scripts/run_baseline.sh)
  if [ -n "$POSTTRAIN_TRAIN_CONFIG" ]; then
    command+=(--base-train "$POSTTRAIN_TRAIN_CONFIG")
  fi
  command+=( \
    --base-dataset  "$POSTTRAIN_DATASET" \
    --base-model    "$model_config" \
    --base-baseline "$baseline" \
    --base-eval     "$POSTTRAIN_EVAL_CONFIG" \
    --seed "$SEED" --run_name "$(basename "$out")" --output_dir "$out" "$@" \
  )
  launch "${command[@]}"
}

# $1 run id, $2 GRPO config, $3 reward config, $4... command-line overrides.
# RUN_MODEL_CONFIG may override the public initialization for a single call.
train_rl_posttrain() {
  local id="$1" grpo="$2" reward="$3"; shift 3
  local model_config="${RUN_MODEL_CONFIG:-$INIT_MODEL_CONFIG}"
  local out; out="$(run_dir "$id" "$model_config")"
  already_done "$out" && return 0
  local command=(bash ./scripts/run.sh)
  if [ -n "$POSTTRAIN_TRAIN_CONFIG" ]; then
    command+=(--base-train "$POSTTRAIN_TRAIN_CONFIG")
  fi
  command+=( \
    --base-dataset "$POSTTRAIN_DATASET" \
    --base-model   "$model_config" \
    --base-grpo    "$grpo" \
    --base-reward  "$reward" \
    --base-eval    "$POSTTRAIN_EVAL_CONFIG" \
    --seed "$SEED" --run_name "$(basename "$out")" --output_dir "$out" "$@" \
  )
  launch "${command[@]}"
}

# Evaluation is post-hoc so MTEB never selects training hyperparameters.
evaluate_model() {
  local model="$1" benchmark="${2:-MTEB(eng, v2)}" model_config="${3:-}"
  if [ -n "$model_config" ]; then
    launch bash eval_mteb/scripts/run_mteb.sh "$model" "$benchmark" "$model_config"
  else
    launch bash eval_mteb/scripts/run_mteb.sh "$model" "$benchmark"
  fi
}
