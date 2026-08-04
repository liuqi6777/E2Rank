#!/bin/bash
# Phase 1 -- Stage 1 plus the two pilots. GATES EVERYTHING.
#
# The pilots run the RANKING TERM ALONE, not the default mixture: what they decide is which
# candidate pool that term should use, and adding the continuous companion would mask exactly
# the degeneracy being measured.
#
# Decision taken at the end of this script, from reward/<term>/n_distinct and
# advantages/degenerate_frac (EXPERIMENT_PLAN.md SS0.2):
#   the 256-candidate pool is worth its cost only if it realizes materially MORE distinct
#   reward levels than the 39-candidate pool. Its extra 217 candidates are other queries'
#   mined negatives, mostly unrelated to this query, so they may never outrank the gold and
#   may enlarge the pool without adding a single reward level.
# If they tie, set DEFAULT_REWARD=configs/reward/default_mixture.yaml (pool 39, the cheaper
# table); otherwise DEFAULT_REWARD=configs/reward/default_mixture_all.yaml.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

require_stage1

# Before either pilot: can the policy reorder these slates at all? A rank-based reward only
# varies when a perturbation flips two candidates, so if the checkpoint's adjacent-rank score
# gaps sit far above g*(kappa), degenerate_frac will be ~1 for reasons no amount of G or kappa
# tuning can fix -- and the two pilots below would then be measuring the same nothing twice.
# Minutes on one GPU, and it also tells you which end of the kappa sweep is worth running.
launch uv run python scripts/measure_score_gaps.py \
  --model "$(stage1_merged_dir)" \
  --data_path "$(uv run python -c "import sys;sys.path.insert(0,'src');from utils import load_raw_config_file;print(load_raw_config_file('configs/dataset/default.yaml')['data_path'])")" \
  --slate_size 8 --num_batches 32

train_rl "pilot-ndcg-ib"     - configs/reward/ndcg_in_batch.yaml     stage1
train_rl "pilot-ndcg-ib-all" - configs/reward/ndcg_in_batch_all.yaml stage1

cat <<'NOTE'

Gate on TRAINING DIAGNOSTICS, not downstream metrics:

  reward/std                     stably > 0                 else the reward is degenerate:
                                                            check the A_d(kappa) rescale and
                                                            the slate ordering first
  advantages/degenerate_frac     < 0.5, not trending up      the 0.2 of the old plan is not
                                                            reachable at these pool sizes
  reward/<term>/n_distinct       compare the two pilots      this is the decision above
  reward/mean                    increases                   else check the log-prob sign and
                                                            the advantage detach
  train/sigma                    constant                    else a config error

Record wall-clock/step and peak memory: every budget estimate in the plan is a multiple of it.
NOTE
