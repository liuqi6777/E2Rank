#!/bin/bash
# Phase A-R (tab:ablation-reward) -- runs R1-R7, the single-term rows.
#
# VERIFICATION, not selection: the default is declared in _common.sh and these rows test it.
# Two predictions are on the line. (i) Each single term loses to the mixture. (ii) The ranking
# rows order by REALIZED reward levels rather than by nominal pool size -- which is why R2 and
# R3 are in the table even though the pilot already ran one of them.
#
# MRR is deliberately absent: under binary single-positive labels its group-standardized
# advantage vector has cosine 0.9992 with nDCG's, so it would duplicate R1. The equivalence is
# argued in the appendix instead, and phaseD_appendix.sh keeps one MRR row as an empirical check.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_stage1

# Rank-based (non-differentiable). Pool / realized levels are the explanatory variable.
train_rl "r1-ndcg"         - configs/reward/ndcg.yaml               stage1   # pool 8
train_rl "pilot-ndcg-ib"   - configs/reward/ndcg_in_batch.yaml      stage1   # pool 39, = pilot
train_rl "pilot-ndcg-ib-all" - configs/reward/ndcg_in_batch_all.yaml stage1  # pool 256, = pilot

# Score-based (differentiable). The theory says these land near their backprop twins.
train_rl "r4-ctr"          - configs/reward/contrastive_no_in_batch.yaml stage1
train_rl "r5-ctr-ib"       - configs/reward/contrastive_in_batch.yaml    stage1
train_rl "r6-infonce"      - configs/reward/infonce_no_in_batch.yaml     stage1
train_rl "r7-infonce-ib"   - configs/reward/infonce_in_batch.yaml        stage1

# R7's backpropagated counterpart is C3 (CL->CL) from phaseB_recipe.sh -- the same objective on
# the same data at the same budget, reaching the encoder directly instead of through the policy
# gradient. It is the one matched pair the differentiability prediction can be tested on, so it
# is not re-trained here; run phaseB_recipe.sh and read the two rows together.
echo
echo "[note] the Delta-vs-backprop row is C3 from phaseB_recipe.sh; no extra run needed."
