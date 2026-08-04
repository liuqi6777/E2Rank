#!/bin/bash
# Phase C (tab:ablation-estimator) -- runs E2, E4-E8. E1/E3 = C7 (reuse).
# Each row is a SINGLE-FLAG diff against configs/grpo/default.yaml.
#
# The frozen-candidate rows need in-batch candidates: the A_d(kappa) rescale is a no-op without
# them. The declared default already carries an in-batch ranking term, so this block runs at the
# default -- if that ever changes to an own-slate reward, these rows must be pinned to an
# in-batch reward explicitly and the caption must say the block deviates from the default.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_stage1

# E2  projected Gaussian at matched mean alignment (A_d(755) = 0.530 = 1/sqrt(1+0.05^2*1024)),
#     so the two sampling laws are compared at equal exploration, not equal nominal parameter.
train_rl "e2-gaussian"        configs/grpo/abl_gaussian.yaml   - stage1

# E4  PREDICTED FAILURE: without the rescale every frozen distractor outranks every sampled
#     document, so reward/std -> 0 and degenerate_frac -> 1 within a few hundred steps. Report
#     the diagnostics even though the run is useless downstream -- they are the claim.
train_rl "e4-no-rescale"      configs/grpo/abl_no_frozen_rescale.yaml - stage1
# E5  the same at kappa=400, where 1/A_d grows from ~1.9 to ~2.9 and the effect should sharpen.
train_rl "e5-no-rescale-k400" configs/grpo/abl_no_frozen_rescale.yaml - stage1 --kappa 400

# E6/E7  advantage normalization.
train_rl "e6-adv-shared"      configs/grpo/abl_adv_shared.yaml - stage1
train_rl "e7-adv-none"        configs/grpo/abl_adv_none.yaml   - stage1

# E8  in-batch candidates scored with sampled rather than detached mean embeddings, which leaks
#     other samples' perturbations into this sample's advantages through the shared group index.
train_rl "e8-ib-sampled"      configs/grpo/abl_in_batch_sampled.yaml - stage1
