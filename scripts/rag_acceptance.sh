#!/usr/bin/env bash
set -euo pipefail

mode="${1:-}"
case "$mode" in
  2gpu-index)
    work_dir="${RAG_TOY_DIR:-/tmp/e2rank-rag-toy}"
    PYTHONPATH=src uv run torchrun --standalone --nproc_per_node=2 \
      scripts/rag_toy_distributed.py --work-dir "$work_dir" --require-cuda
    ;;
  4gpu-e2e)
    PYTHONPATH=src uv run torchrun --standalone --nproc_per_node=4 src/train_rag.py \
      configs/rag/infonce.yaml \
      --rag_split full --rag_max_train_samples 128 --max_steps 2 \
      --save_strategy no --report_to none \
      --output_dir checkpoints/rag-4gpu-smoke --overwrite_output_dir true
    ;;
  *)
    echo "usage: scripts/rag_acceptance.sh {2gpu-index|4gpu-e2e}" >&2
    exit 2
    ;;
esac
