#!/bin/bash
# Post-hoc evaluation of every checkpoint under CKPT_ROOT.
#
#   bash scripts/experiments/eval_all.sh full     # MTEB(eng, v2)      -- main + recipe rows
#   bash scripts/experiments/eval_all.sh subset   # MTEB(eng, v1, subset) -- ablations
#   bash scripts/experiments/eval_all.sh bright   # BRIGHT
#
# Restrict with:  CKPT_GLOB='checkpoints/c7-*' bash scripts/experiments/eval_all.sh full
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

MODE="${1:-subset}"
case "$MODE" in
  full)   BENCHMARK="MTEB(eng, v2)" ;;
  subset) BENCHMARK="MTEB(eng, v1, subset)" ;;
  bright) BENCHMARK="BRIGHT" ;;
  *) echo "Usage: $0 [full|subset|bright]" >&2; exit 1 ;;
esac

CKPT_GLOB="${CKPT_GLOB:-${CKPT_ROOT}/*}"
for ckpt in $CKPT_GLOB; do
  [ -d "$ckpt" ] || continue
  case "$ckpt" in *-merged) continue ;; esac   # intermediates, not results
  launch bash eval_mteb/scripts/run_mteb.sh "$ckpt" "$BENCHMARK"
done

echo
echo "Summarize with:"
echo "  python eval_mteb/summary.py results/mteb/<model>/<label>/no_version_available \"$BENCHMARK\""
