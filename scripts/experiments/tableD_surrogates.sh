#!/bin/bash
# Appendix Table D -- differentiable ranking surrogates as the Stage-2 objective (D1-D5),
# preceded by the smoothing sweep on the dev split.
#
#   SWEEP=1 bash scripts/experiments/tableD_surrogates.sh   # tune first
SEEDS="${SEEDS:-42}"
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
prepare_stage1

if [ "${SWEEP:-0}" = "1" ]; then
  # Selection runs. Scored on the held-out dev split (dev_samples_per_source in the
  # dataset config), never on MTEB: tuning a smoothing parameter on the benchmark the
  # paper reports would invalidate every number in it.
  for seed in $SEEDS; do
    for v in 0.5 1 2;  do train_supervised_stage2 "sweep-ranknet-s${v}"    "$seed" configs/baseline/ranknet.yaml    "${DEV_EVAL_FLAGS[@]}" --ranknet_sigma "$v"; done
    for v in 0.5 1 2;  do train_supervised_stage2 "sweep-lambdaloss-s${v}" "$seed" configs/baseline/lambdaloss.yaml "${DEV_EVAL_FLAGS[@]}" --lambdaloss_sigma "$v"; done
    for v in 5 10 20;  do train_supervised_stage2 "sweep-approxndcg-a${v}" "$seed" configs/baseline/approxndcg.yaml "${DEV_EVAL_FLAGS[@]}" --approxndcg_alpha "$v"; done
    for v in 0.5 1 2;  do train_supervised_stage2 "sweep-softrank-s${v}"   "$seed" configs/baseline/softrank.yaml   "${DEV_EVAL_FLAGS[@]}" --softrank_sigma "$v"; done
    for v in 0.5 1 2;  do train_supervised_stage2 "sweep-neuralndcg-t${v}" "$seed" configs/baseline/neuralndcg.yaml "${DEV_EVAL_FLAGS[@]}" --neuralndcg_temperature "$v"; done
  done
  echo
  echo "Sweep done. Select the max eval/ndcg per surrogate, record the grid AND the spread"
  echo "in Appendix Table 12, then re-run without SWEEP=1 passing the chosen values."
  exit 0
fi

for seed in $SEEDS; do
  train_supervised_stage2 "d1-ranknet"    "$seed" configs/baseline/ranknet.yaml
  train_supervised_stage2 "d2-lambdaloss" "$seed" configs/baseline/lambdaloss.yaml
  train_supervised_stage2 "d3-approxndcg" "$seed" configs/baseline/approxndcg.yaml
  train_supervised_stage2 "d4-softrank"   "$seed" configs/baseline/softrank.yaml
  train_supervised_stage2 "d5-neuralndcg" "$seed" configs/baseline/neuralndcg.yaml
done
