#!/bin/bash
# Table 2 (tab:recipe) -- runs C1-C8, the controlled comparison at 0.6B.
# This is where every causal claim lives; run it before anything else.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# C1 needs no training: the base LLM is evaluated as-is.
echo "[C1] base LLM, no training -- eval only ($BASE_MODEL_ID)"

# C2 (CL-only) is the single shared Stage 1.
prepare_stage1

# --- Pivotal rows: C3 / C7 at PIVOT_SEEDS (Stage-2 seeds only) ---------------------
# Both branch from the SAME Stage-1 checkpoint, so this is a paired comparison: the two
# rows differ only in the Stage-2 objective. The spread measured here is the noise floor
# quoted in every ablation caption, and is conditional on that Stage-1 checkpoint.
for seed in $PIVOT_SEEDS; do
  train_supervised_stage2 "c3-clcl" "$seed" configs/baseline/infonce.yaml
  train_rl "c7-clrl" "$seed" configs/grpo/grid.yaml configs/reward/ndcg.yaml stage1
done

# --- Remaining recipe rows: single seed --------------------------------------------
for seed in $SEEDS; do

  # C4  RL-only: no Stage 1 at all.
  train_rl "c4-rlonly"    "$seed" configs/grpo/grid.yaml   configs/reward/ndcg.yaml base

  # C5  RL-only + vMF KL anchor against the LoRA-disabled base.
  train_rl "c5-rlonly-kl" "$seed" configs/grpo/abl_kl.yaml configs/reward/ndcg.yaml base

  # C6  CL->REINFORCE: group baseline replaced by a global running one.
  train_rl "c6-reinforce" "$seed" configs/grpo/abl_reinforce.yaml configs/reward/ndcg.yaml stage1
done

# C8  short Stage 1 (10% of its steps), then the same RL stage. Needs its own Stage-1
#     checkpoint -- also a single seed -- so it sits outside the loops above.
echo
echo "[C8] short Stage 1 -> RL. Set STAGE1_MAX_STEPS to 10% of the Stage-1 step count."
if [ -n "${STAGE1_MAX_STEPS:-}" ]; then
  out="${CKPT_ROOT}/stage1short-${SCALE}-s${STAGE1_SEED}"
  already_done "$out" || launch bash ./scripts/run_baseline.sh \
    --base-train configs/train/stage1.yaml --base-dataset configs/dataset/stage1.yaml \
    --base-model "$BASE_MODEL_CONFIG" --base-baseline configs/baseline/infonce.yaml \
    --max_steps "$STAGE1_MAX_STEPS" \
    --seed "$STAGE1_SEED" --run_name "$(basename "$out")" --output_dir "$out"
  merged="${out}-merged"
  already_done "$merged" || launch uv run python scripts/merge_lora.py \
    --base "$BASE_MODEL_ID" --adapter "$out" --out "$merged" --overwrite
  for seed in $SEEDS; do
    rl_out="$(run_dir "c8-shortcl-clrl" "$seed")"
    already_done "$rl_out" || launch bash ./scripts/run.sh \
      --base-train configs/train/stage2.yaml --base-dataset configs/dataset/stage2.yaml \
      --base-model "$BASE_MODEL_CONFIG" --model_name_or_path "$merged" \
      --base-grpo configs/grpo/grid.yaml --base-reward configs/reward/ndcg.yaml \
      --base-eval configs/eval/default.yaml \
      --seed "$seed" --run_name "$(basename "$rl_out")" --output_dir "$rl_out"
  done
else
  echo "  [skip] STAGE1_MAX_STEPS unset"
fi
