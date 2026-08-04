#!/bin/bash
# Phase A-M (tab:ablation-mixture) -- runs M1-M6.
#
# Runs FIRST among the sweeps, because the mixture weight w is the one parameter the argument
# in SS0.3 does not fix. Everything else starts at w=0.5 and is corrected with a handful of
# re-runs if this disagrees -- a far better trade than idling a node behind a sweep.
#
# The table's two endpoint rows are not run here: they are R2/R3 and R7 from phaseA_reward.sh.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_stage1

# M1-M4: the weight. M2 (w=0.5) IS the declared default, so it is named for what it is and
# reused by every other script rather than re-trained here.
train_rl "m1-mix-w0.25"  - configs/reward/mixture_w0.25.yaml     stage1
train_rl "c7-clrl"       - "$DEFAULT_REWARD"                     stage1   # M2 = the default
train_rl "m3-mix-w1"     - configs/reward/mixture_w1.yaml        stage1
train_rl "m4-mix-w2"     - configs/reward/mixture_w2.yaml        stage1

# M5: the SAME terms and weight combined with a raw sum. Predicted to collapse to the InfoNCE
# term, because under 'sum' the mixing ratio is weight x within-group std and the two families'
# spreads differ by ~an order of magnitude. Read reward/<term>/group_std for the ratio obtained.
# Re-run at the selected weight if M1-M4 do not choose 0.5.
train_rl "m5-mix-raw-sum" - configs/reward/mixture_raw_sum.yaml    stage1

# M6: the other continuous family, to check the mixture is about mixing families rather than
# about InfoNCE in particular.
train_rl "m6-mix-ctr"     - configs/reward/mixture_contrastive.yaml stage1
