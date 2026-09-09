#!/bin/bash
# Shared helpers for the experiment scripts. Source this; do not run it.
#
#   SEED=42            the single seed everything runs at
#   DRY_RUN=1          print commands without launching
#   SCALE=0.6b|4b|8b   which base LLM (default 0.6b)
#   CKPT_ROOT=...      where checkpoints land (default ./checkpoints)
#   DEFAULT_REWARD=... override the default reward config (see below)
#
# Run order (EXPERIMENT_PLAN.md SS3):
#   stage1.sh             the shared contrastive checkpoint. Run once, first, per scale.
#   phase1_pilot.sh       the two pilots that fix the nDCG pool. GATES EVERYTHING.
#   reward_probe.sh       free: reward signal vs. Stage-1 progress. Trains nothing.
#   phaseA_mixture.sh     the mixture weight sweep; fixes w. Short serial dependency.
#   phaseB_recipe.sh      the controlled comparison -- the rows that decide the paper.
#   phaseA_reward.sh      \
#   phaseA_components.sh   > verification + ablations; all run at the declared default, in
#   phaseC_estimator.sh    > parallel with each other and with Phase B.
#   phaseD_appendix.sh    /
#   phaseE_scale.sh       4B / 8B. First to cut.
#   eval_all.sh           post-hoc MTEB.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Seed policy: ONE seed, ONE Stage-1 checkpoint that every CL->* row branches from. Differences
# between rows are therefore attributable to the changed setting, but no dispersion is measured
# and none is claimed -- see the single-seed limitation in the appendix.
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-0}"
SCALE="${SCALE:-0.6b}"
CKPT_ROOT="${CKPT_ROOT:-checkpoints}"
export WANDB_PROJECT="${WANDB_PROJECT:-E2Rank-RL}"

# The default reward is DECLARED, not selected (EXPERIMENT_PLAN.md SS2.1): the in-batch-extended
# ranking term plus a continuous InfoNCE companion. Two knobs are settled by experiment, and
# both are one line here rather than an edit inside a config:
#   - the pilot decides the ranking term's pool -> flip to default_mixture_all.yaml if the
#     256-candidate pool realizes materially more reward levels;
#   - phaseA_mixture.sh decides w               -> everything runs at w=0.5 until it lands.
DEFAULT_REWARD="${DEFAULT_REWARD:-configs/reward/default_mixture.yaml}"
DEFAULT_GRPO="${DEFAULT_GRPO:-configs/grpo/default.yaml}"
DEFAULT_DATASET="${DEFAULT_DATASET:-configs/dataset/stage2.yaml}"
DEFAULT_BASELINE="${DEFAULT_BASELINE:-configs/baseline/default.yaml}"

case "$SCALE" in
  0.6b) BASE_MODEL_CONFIG=configs/model/qwen3_0.6b.yaml; BASE_MODEL_ID=Qwen/Qwen3-0.6B ;;
  4b)   BASE_MODEL_CONFIG=configs/model/qwen3_4b.yaml;   BASE_MODEL_ID=Qwen/Qwen3-4B   ;;
  8b)   BASE_MODEL_CONFIG=configs/model/qwen3_8b.yaml;   BASE_MODEL_ID=Qwen/Qwen3-8B   ;;
  *)    echo "Unknown SCALE=$SCALE (expected 0.6b, 4b, or 8b)" >&2; exit 1 ;;
esac

stage1_dir()              { echo "${CKPT_ROOT}/stage1-${SCALE}-s${SEED}"; }
stage1_merged_dir()       { echo "${CKPT_ROOT}/stage1-${SCALE}-s${SEED}-merged"; }
run_dir()                 { echo "${CKPT_ROOT}/${1}-${SCALE}-s${SEED}"; }

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
# Trained once; every CL->* row branches from it. Same corpus as Stage 2, so the recipe
# comparison carries no data confound at all.
train_stage1() {
  local out; out="$(stage1_dir)"
  already_done "$out" && return 0
  launch bash ./scripts/run_baseline.sh \
    --base-train    configs/train/stage1.yaml \
    --base-dataset  configs/dataset/stage1.yaml \
    --base-model    "$BASE_MODEL_CONFIG" \
    --base-baseline configs/baseline/default.yaml \
    --base-eval     configs/eval/default.yaml \
    --seed "$SEED" --run_name "$(basename "$out")" --output_dir "$out" "$@"
}

# --- Merge Stage 1 so Stage 2 starts from a standalone checkpoint --------------------
# Merging rather than chaining via lora_path is required for the KL anchor: it is taken by
# disabling the adapter, so the reference must be the Stage-1 model, not the raw base LLM.
merge_stage1() {
  local src merged; src="$(stage1_dir)"; merged="$(stage1_merged_dir)"
  already_done "$merged" && return 0
  launch uv run python scripts/merge_lora.py \
    --base "$BASE_MODEL_ID" --adapter "$src" --out "$merged" --overwrite
}

# --- Stage 2: ranking RL ------------------------------------------------------------
# $1 run id   $2 grpo config   $3 reward config   $4 init (base|stage1)   $5... overrides
# Pass "-" in the grpo or reward slot to take the declared default. RL_DATASET / RL_TRAIN
# swap the dataset / train slot for the rows that need it.
train_rl() {
  local id="$1" grpo="$2" reward="$3" init="$4"; shift 4
  [ "$grpo"   = "-" ] && grpo="$DEFAULT_GRPO"
  [ "$reward" = "-" ] && reward="$DEFAULT_REWARD"
  local dataset="${RL_DATASET:-$DEFAULT_DATASET}"
  local train_cfg="${RL_TRAIN:-configs/train/stage2.yaml}"
  local out; out="$(run_dir "$id")"
  already_done "$out" && return 0
  local model_flag=(--base-model "$BASE_MODEL_CONFIG")
  if [ "$init" = "stage1" ]; then
    model_flag+=(--model_name_or_path "$(stage1_merged_dir)")
  fi
  launch bash ./scripts/run.sh \
    --base-train   "$train_cfg" \
    --base-dataset "$dataset" \
    "${model_flag[@]}" \
    --base-grpo    "$grpo" \
    --base-reward  "$reward" \
    --base-eval    configs/eval/default.yaml \
    --seed "$SEED" --run_name "$(basename "$out")" --output_dir "$out" "$@"
}

# --- Stage 2 with the supervised objective (the CL->CL control) ----------------------
# $1 run id   $2... overrides
train_supervised_stage2() {
  local id="$1"; shift
  local baseline="${BASELINE_CONFIG:-$DEFAULT_BASELINE}"
  local dataset="${SUPERVISED_DATASET:-$DEFAULT_DATASET}"
  local out; out="$(run_dir "$id")"
  already_done "$out" && return 0
  launch bash ./scripts/run_baseline.sh \
    --base-train    configs/train/stage2.yaml \
    --base-dataset  "$dataset" \
    --base-model    "$BASE_MODEL_CONFIG" \
    --base-baseline "$baseline" \
    --base-eval     configs/eval/default.yaml \
    --model_name_or_path "$(stage1_merged_dir)" \
    --seed "$SEED" --run_name "$(basename "$out")" --output_dir "$out" "$@"
}

# --- Stage-1 dependency -------------------------------------------------------------
# The phase scripts REQUIRE Stage 1; they do not train it. It is the longest job in the plan
# and shared by everything, so starting it as a side effect of launching a 1.5-hour ablation is
# exactly the accident worth preventing. stage1.sh is the only place it is trained.
require_stage1() {
  local merged; merged="$(stage1_merged_dir)"
  if [ -f "$merged/config.json" ] || [ "$DRY_RUN" = "1" ]; then return 0; fi
  cat >&2 <<EOF

Stage-1 checkpoint missing: $merged

Train it once, then re-run this script:
  SCALE=$SCALE bash scripts/experiments/stage1.sh

EOF
  exit 1
}
