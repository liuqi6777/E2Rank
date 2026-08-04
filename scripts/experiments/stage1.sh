#!/bin/bash
# Stage 1 -- the contrastive (InfoNCE) checkpoint EVERY CL->* row in the paper branches from.
# All Stage-1 training lives here and nowhere else.
#
#   bash scripts/experiments/stage1.sh              # the shared Stage-1 checkpoint
#   SCALE=4b bash scripts/experiments/stage1.sh     # the 4B/8B rows of tab:main-mteb
#
# Its INTERMEDIATE checkpoints are not scratch: reward_probe.sh reads them to measure how much
# contrastive training the ranking reward needs before it carries signal. Keep save_steps small
# enough that the curve has points on it (configs/train/stage1.yaml).
#
# Why this is its own script. It is the longest single job in the plan, it is shared by every
# downstream row, and it is the one run that must NOT be started by accident: the phase scripts
# therefore *require* its output rather than training it themselves, and fail with a pointer
# back here if it is missing. Run this once per scale, then branch.
#
# Two properties of the output are load-bearing downstream:
#   - it trains on the SAME corpus as Stage 2, so the recipe comparison in phaseB_recipe.sh has
#     no data confound -- only the objective differs;
#   - it is MERGED into standalone weights afterwards. Chaining via lora_path would leave the
#     KL anchor referencing the raw base LLM, since the anchor is taken by disabling the
#     adapter; merging makes the reference the Stage-1 model, which is what the method assumes.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

train_stage1
merge_stage1

echo
echo "Stage 1 ready at $(stage1_merged_dir). Everything downstream branches from it:"
echo "  bash scripts/experiments/phase1_pilot.sh     # gate on the diagnostics first"
echo "  bash scripts/experiments/reward_probe.sh     # free: reward signal vs. Stage-1 progress"
