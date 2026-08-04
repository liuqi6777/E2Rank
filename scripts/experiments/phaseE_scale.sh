#!/bin/bash
# Phase E (tab:main-mteb) -- the 4B / 8B rows. 0.6B reuses C2 (Stage 1) and C7.
#
#   SCALE=4b bash scripts/experiments/phaseE_scale.sh
#   SCALE=8b bash scripts/experiments/phaseE_scale.sh
#
# THE FIRST THING TO CUT if compute is short. The expensive part is the 4B/8B STAGE-1 runs, not
# the RL stages, and the paper's claims live in phaseB_recipe.sh: a 0.6B-only paper with a clean
# controlled comparison is stronger than a three-scale paper with a rushed one. If cut, say
# "single scale" in Limitations rather than implying scale was tested.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

if [ "$SCALE" = "0.6b" ]; then
  echo "[note] 0.6B rows are C2 (Stage 1) and C7 from phaseB_recipe.sh -- nothing to do here."
  echo "       Run with SCALE=4b or SCALE=8b."
  exit 0
fi

# The CL-only reference row for this scale IS Stage 1 at this scale; train it with
#   SCALE=$SCALE bash scripts/experiments/stage1.sh
require_stage1
train_rl "m-clrl" - - stage1

echo
echo "Evaluate with:  bash scripts/experiments/eval_all.sh full"
