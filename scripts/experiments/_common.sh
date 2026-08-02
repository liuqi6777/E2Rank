#!/bin/bash
# Shared helpers for the experiment scripts. Source this; do not run it.
#
#   SEEDS             Stage-2 seeds for ordinary rows (default: 42)
#   PIVOT_SEEDS       Stage-2 seeds for the pivotal CL->CL vs CL->RL rows (default: 42 43 44)
#   STAGE1_SEED       the single Stage-1 seed everything branches from (default: 42)
#   DRY_RUN=1         print commands without launching
#   SCALE=0.6b|4b|8b  which base LLM (default 0.6b)
#   CKPT_ROOT=...     where checkpoints land (default ./checkpoints)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Seed policy.
#
# ONE Stage-1 checkpoint, and every downstream row branches from it. Seeds vary only in
# Stage 2. This makes the pivotal comparison a *paired* design: C3 and C7 differ solely in
# the Stage-2 objective, with Stage-1 variance held fixed rather than added as noise, which
# is exactly the contrast the paper claims. What it does NOT measure is how much the result
# would move if Stage 1 were retrained -- so the sd is reported as "conditional on the
# Stage-1 checkpoint", never as full-pipeline variance.
#
# Everything outside the pivotal rows runs at a single seed and is read against the sd
# measured there; the noise floor is a property of the setup, not of each ablation.
SEEDS="${SEEDS:-42}"
PIVOT_SEEDS="${PIVOT_SEEDS:-42 43 44}"
STAGE1_SEED="${STAGE1_SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
SCALE="${SCALE:-0.6b}"
CKPT_ROOT="${CKPT_ROOT:-checkpoints}"
export WANDB_PROJECT="${WANDB_PROJECT:-E2Rank-RL}"

case "$SCALE" in
  0.6b) BASE_MODEL_CONFIG=configs/model/qwen3_0.6b.yaml; BASE_MODEL_ID=Qwen/Qwen3-0.6B ;;
  4b)   BASE_MODEL_CONFIG=configs/model/qwen3_4b.yaml;   BASE_MODEL_ID=Qwen/Qwen3-4B   ;;
  8b)   BASE_MODEL_CONFIG=configs/model/qwen3_8b.yaml;   BASE_MODEL_ID=Qwen/Qwen3-8B   ;;
  *)    echo "Unknown SCALE=$SCALE (expected 0.6b, 4b, or 8b)" >&2; exit 1 ;;
esac

# Every ablation shares one Stage-1 checkpoint per (scale, seed). Train it once; branch after.
stage1_dir()        { echo "${CKPT_ROOT}/stage1-${SCALE}-s${STAGE1_SEED}"; }
stage1_merged_dir() { echo "${CKPT_ROOT}/stage1-${SCALE}-s${STAGE1_SEED}-merged"; }
run_dir()           { echo "${CKPT_ROOT}/${1}-${SCALE}-s${2}"; }

log() { printf '\n\033[1m>>> %s\033[0m\n' "$*"; }

launch() {
  log "$*"
  if [ "$DRY_RUN" = "1" ]; then return 0; fi
  "$@"
}

# Skip work that is already on disk, so a failed sweep can be resumed by re-running it.
already_done() {
  if [ -f "$1/config.json" ] || [ -f "$1/adapter_config.json" ]; then
    echo "  [skip] $1 already exists"
    return 0
  fi
  return 1
}

# --- Stage 1: contrastive (InfoNCE) from the base LLM -------------------------------
# Trained once at STAGE1_SEED; every CL->* row branches from it.
train_stage1() {
  local out; out="$(stage1_dir)"
  already_done "$out" && return 0
  launch bash ./scripts/run_baseline.sh \
    --base-train    configs/train/stage1.yaml \
    --base-dataset  configs/dataset/stage1.yaml \
    --base-model    "$BASE_MODEL_CONFIG" \
    --base-baseline configs/baseline/infonce.yaml \
    --seed "$STAGE1_SEED" --run_name "$(basename "$out")" --output_dir "$out" "$@"
}

# --- Merge Stage 1 so Stage 2 starts from a standalone checkpoint --------------------
merge_stage1() {
  local src merged; src="$(stage1_dir)"; merged="$(stage1_merged_dir)"
  already_done "$merged" && return 0
  launch uv run python scripts/merge_lora.py \
    --base "$BASE_MODEL_ID" --adapter "$src" --out "$merged" --overwrite
}

# --- Stage 2: ranking RL ------------------------------------------------------------
# $1 run id   $2 seed   $3 grpo config   $4 reward config   $5 init (base|stage1)   $6... overrides
train_rl() {
  local id="$1" seed="$2" grpo="$3" reward="$4" init="$5"; shift 5
  local out; out="$(run_dir "$id" "$seed")"
  already_done "$out" && return 0
  local model_flag=()
  if [ "$init" = "stage1" ]; then
    model_flag=(--base-model "$BASE_MODEL_CONFIG" --model_name_or_path "$(stage1_merged_dir)")
  else
    model_flag=(--base-model "$BASE_MODEL_CONFIG")
  fi
  launch bash ./scripts/run.sh \
    --base-train   configs/train/stage2.yaml \
    --base-dataset configs/dataset/stage2.yaml \
    "${model_flag[@]}" \
    --base-grpo    "$grpo" \
    --base-reward  "$reward" \
    --base-eval    configs/eval/default.yaml \
    --seed "$seed" --run_name "$(basename "$out")" --output_dir "$out" "$@"
}

# Dev-set evaluation flags, used by the smoothing sweep. The split itself is configured in
# configs/dataset/stage2*.yaml (dev_samples_per_source); these turn evaluation on.
DEV_EVAL_FLAGS=(--eval_strategy steps --eval_steps 100 --per_device_eval_batch_size 32
                --save_strategy no --report_to wandb)

# --- Stage 2 with a supervised objective (the CL->CL control and the surrogates) -----
# $1 run id   $2 seed   $3 baseline config   $4... overrides
train_supervised_stage2() {
  local id="$1" seed="$2" baseline="$3"; shift 3
  local out; out="$(run_dir "$id" "$seed")"
  already_done "$out" && return 0
  launch bash ./scripts/run_baseline.sh \
    --base-train    configs/train/stage2.yaml \
    --base-dataset  configs/dataset/stage2.yaml \
    --base-model    "$BASE_MODEL_CONFIG" \
    --base-baseline "$baseline" \
    --model_name_or_path "$(stage1_merged_dir)" \
    --seed "$seed" --run_name "$(basename "$out")" --output_dir "$out" "$@"
}

# Ensure the single shared Stage-1 checkpoint exists. Idempotent across scripts.
prepare_stage1() {
  train_stage1
  merge_stage1
}
