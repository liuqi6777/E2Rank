#!/bin/bash
# Phase B (tab:recipe) -- runs C1-C7, the controlled comparison at 0.6B.
#
# THIS IS WHERE EVERY CAUSAL CLAIM LIVES. All rows start from the same base LLM, see the SAME
# corpus in both stages, and consume the same total step budget, so the only variable is how
# that budget is spent. C3 vs C7 is the pair that decides whether the paper exists.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# C1 needs no training: the base LLM is evaluated as-is.
echo "[C1] base LLM, no training -- eval only ($BASE_MODEL_ID)"

# C2 (CL-only) IS the shared Stage 1 -- trained by stage1.sh, evaluated as a row here.
require_stage1

# C3  CL->CL: the compute-matched control, and the row a skeptic reads first. Also serves as
#     the backpropagated counterpart of R7 (InfoNCE as a reward) in tab:ablation-reward.
train_supervised_stage2 "c3-clcl"

# C7  CL->RL at the declared default. Trained by phaseA_mixture.sh under the same run id;
#     whichever script runs first trains it and the other skips.
train_rl "c7-clrl"       -                              - stage1

# C4  RL-only: no Stage 1 at all. SS0.2 predicts this is the hard case -- from an arbitrary
#     initial ordering the ranking term is near-constant across rollouts. Watch degen. frac.
#     from step 0; if it is near 1 immediately, that is the mechanism, and it is a reportable
#     boundary of the method rather than a failed run.
train_rl "c4-rlonly"     -                              - base

# C5  RL-only + vMF KL anchor against the LoRA-disabled base.
train_rl "c5-rlonly-kl"  configs/grpo/abl_kl.yaml       - base

# C6  CL->REINFORCE: the group baseline replaced by a global running one. Single-term reward
#     only -- 'ema' cannot baseline several reward scales with one scalar, so this row uses the
#     ranking term alone and the caption must say so.
train_rl "c6-reinforce"  configs/grpo/abl_reinforce.yaml configs/reward/ndcg_in_batch.yaml stage1

# C8 (short Stage 1) was CUT. It existed to locate how little contrastive training the ranking
# reward needs before it becomes informative -- but it answered that with ONE interior point
# between C4 (no Stage 1) and C7 (full Stage 1), and it cost a whole extra Stage-1 run to do it.
# reward_probe.sh measures the same thing as a curve, over the Stage-1 checkpoints that are
# already on disk, without training anything. See EXPERIMENT_PLAN.md SS0.6.
echo
echo "[note] the Stage-1-sufficiency question is answered by reward_probe.sh, not by a run here."
