#!/usr/bin/env bash
set -euo pipefail

command_name="${1:-}"
shift || true

case "$command_name" in
  prepare)
    PYTHONPATH=src uv run python -m rag.prepare "$@"
    ;;
  encode)
    PYTHONPATH=src uv run torchrun --standalone --nproc_per_node=4 -m rag.encode_corpus "$@"
    ;;
  candidates)
    PYTHONPATH=src uv run python -m rag.mine_candidates "$@"
    ;;
  train)
    config="${1:?usage: scripts/rag_pipeline.sh train CONFIG [overrides]}"
    shift
    PYTHONPATH=src uv run torchrun --standalone --nproc_per_node=4 src/train_rag.py "$config" "$@"
    ;;
  eval)
    PYTHONPATH=src uv run python src/eval_rag.py "$@"
    ;;
  tune-eval)
    PYTHONPATH=src uv run python src/eval_rag_tuning.py "$@"
    ;;
  *)
    echo "usage: scripts/rag_pipeline.sh {prepare|encode|candidates|train|tune-eval|eval} ..." >&2
    exit 2
    ;;
esac
