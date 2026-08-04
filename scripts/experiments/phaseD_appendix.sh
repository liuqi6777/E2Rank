#!/bin/bash
# Phase D -- the appendix ablations: slate size, group size / rollout, kappa, and the secondary
# single-flag rows. All are substitutions into the Stage-2 slot at the declared default.
#
#   bash scripts/experiments/phaseD_appendix.sh          # everything
#   bash scripts/experiments/phaseD_appendix.sh slate    # one block
# Blocks: slate | group | kappa | secondary
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_stage1

BLOCK="${1:-all}"
want() { [ "$BLOCK" = "all" ] || [ "$BLOCK" = "$1" ]; }

# --- Slate size (tab:ablation-slate) ------------------------------------------------
# The highest-value table in the appendix: the direct test of SS0.2. Under binary
# single-positive labels the slate is the ranking reward's RESOLUTION, not only the task's
# difficulty. Prediction: degen. frac. falls monotonically with n. Unlike the in-batch pools
# this lever costs encoder time, linearly in n.
if want slate; then
  for n in 16 32; do
    RL_DATASET="configs/dataset/stage2_slate${n}.yaml" \
      train_rl "s-slate${n}" - - stage1
  done
fi

# --- Group size and rollout structure (tab:ablation-group-size) ---------------------
# G=32 product is C7. Record peak memory per row.
if want group; then
  for g in 4 8 16 64; do
    train_rl "g-product-${g}"  -                            - stage1 --group_size "$g"
  done
  for g in 32 64; do
    train_rl "g-diagonal-${g}" configs/grpo/abl_diagonal.yaml - stage1 --group_size "$g"
  done
fi

# --- Exploration scale (tab:ablation-kappa) -----------------------------------------
# Report A_d(kappa) alongside kappa: it, not kappa, is the interpretable axis.
if want kappa; then
  for k in 200 400 1500 4000; do
    train_rl "k-${k}" - - stage1 --kappa "$k"
  done
  train_rl "k-learnable" configs/grpo/abl_sigma_learnable.yaml - stage1
fi

# --- Secondary single-flag rows (tab:additional-ablations) --------------------------
if want secondary; then
  # X1  MRR: the one empirical check on the SS0.1 algebra that says it duplicates nDCG.
  #     Cut this first if compute is tight.
  train_rl "x1-mrr"       - configs/reward/mrr.yaml  stage1

  # X2-X4  reward cutoff on the ranking term. Swept through configs, NOT --reward_ndcg_k: a
  #        term that names its own k ignores the run-level default, so the CLI override this
  #        used to pass was silently a no-op.
  #        The uncapped row is the direction that matters. A cutoff manufactures ties -- every
  #        rollout that drops the gold below rank k collapses onto reward 0 -- so k+1 bounds the
  #        levels (17 at @16 over the 39-pool, versus 39 uncapped). Read it against
  #        reward/<term>/n_distinct, not just the metric.
  for k in 2 4 _uncapped; do
    train_rl "x-ndcgk${k}" - "configs/reward/mixture_k${k}.yaml" stage1
  done

  # X4  teacher-score grades on the mined negatives -- a different supervision claim, not a
  #     label-scheme detail. One row, explicitly not the default.
  RL_DATASET=configs/dataset/stage2_graded.yaml train_rl "x4-graded" - - stage1

  # X5-X6  KL anchor.
  for b in 0.001 0.01; do
    train_rl "x-kl-${b}" configs/grpo/abl_kl.yaml - stage1 --kl_coef "$b"
  done

  # X7  two epochs.
  RL_TRAIN=configs/train/stage2_2ep.yaml train_rl "x7-2ep" - - stage1

  # X8  full fine-tuning instead of LoRA. kl_coef > 0 is unavailable here: the KL anchor is
  #     taken by disabling an adapter and there is none.
  RL_TRAIN=configs/train/stage2_full_ft.yaml train_rl "x8-full-ft" - - stage1
fi
