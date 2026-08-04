#!/bin/bash
# Reward signal vs. contrastive progress -- the replacement for the cut C8 row.
#
# The question C8 asked was "how much contrastive training does the ranking reward need before
# it carries signal?", and it answered with ONE interior point between C4 (no Stage 1) and C7
# (full Stage 1), bought with an entire extra Stage-1 run. This measures the same thing as a
# CURVE, over the intermediate checkpoints stage1.sh already wrote, and trains nothing.
#
#   bash scripts/experiments/reward_probe.sh
#   PROBE_STEPS=100 bash scripts/experiments/reward_probe.sh
#
# --learning_rate 0 is the point, not an optimization: with no parameter update the diagnostics
# are a property of the CHECKPOINT rather than of N steps of adaptation on top of it, so the
# curve says what it claims to say. Everything else -- reward, policy, data, seed -- is the
# declared default, so a point here is comparable to the C4 and C7 rows that bracket it.
#
# Read three series against the Stage-1 step count:
#   reward/std                       does the group spread exist at all?
#   advantages/degenerate_frac       what fraction of groups carry no gradient?
#   reward/<term>/n_distinct         how many reward levels does a group actually resolve?
# The prediction is that all three are flat and useless at step 0 -- an untrained encoder orders
# slates arbitrarily, so the ranking term is near-constant across rollouts -- and that they turn
# over well before Stage 1 finishes. Where they turn over IS the answer; if they never do, the
# ranking reward is not a second-stage objective but a non-starter, and that is the finding.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

PROBE_STEPS="${PROBE_STEPS:-50}"
STAGE1="$(stage1_dir)"

if [ ! -d "$STAGE1" ] && [ "$DRY_RUN" != "1" ]; then
  echo "No Stage-1 run at $STAGE1. Train it first: bash scripts/experiments/stage1.sh" >&2
  exit 1
fi

# The base LLM itself is the step-0 point of the curve, and it is also C4's initialization.
train_rl "probe-step0" - - base --max_steps "$PROBE_STEPS" --learning_rate 0 --save_strategy no

shopt -s nullglob
# Guarded expansion below: under `set -u`, bash 3.2 (the macOS default) treats "${arr[@]}" on an
# empty array as an unbound variable rather than as zero words.
checkpoints=("$STAGE1"/checkpoint-*)
if [ ${#checkpoints[@]} -eq 0 ]; then
  echo
  echo "  [note] no intermediate checkpoints under $STAGE1 -- only the step-0 and full-Stage-1"
  echo "         ends of the curve are available. Lower save_steps in configs/train/stage1.yaml"
  echo "         and retrain if the curve needs interior points."
fi

for ckpt in ${checkpoints[@]+"${checkpoints[@]}"}; do
  step="$(basename "$ckpt" | sed 's/checkpoint-//')"
  merged="${ckpt}-merged"
  already_done "$merged" || launch uv run python scripts/merge_lora.py \
    --base "$BASE_MODEL_ID" --adapter "$ckpt" --out "$merged" --overwrite
  out="$(run_dir "probe-step${step}")"
  already_done "$out" || launch bash ./scripts/run.sh \
    --base-train "configs/train/stage2.yaml" --base-dataset "$DEFAULT_DATASET" \
    --base-model "$BASE_MODEL_CONFIG" --model_name_or_path "$merged" \
    --base-grpo "$DEFAULT_GRPO" --base-reward "$DEFAULT_REWARD" \
    --base-eval configs/eval/default.yaml \
    --max_steps "$PROBE_STEPS" --learning_rate 0 --save_strategy no \
    --seed "$SEED" --run_name "$(basename "$out")" --output_dir "$out"
done

# The far end of the curve is C7's initialization, i.e. the merged full Stage-1 checkpoint.
require_stage1
train_rl "probe-stage1-full" - - stage1 --max_steps "$PROBE_STEPS" --learning_rate 0 --save_strategy no
