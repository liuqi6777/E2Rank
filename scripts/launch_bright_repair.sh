#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/launch_bright_repair.sh SLOT [extra launcher options]" >&2
  echo "Run once on each of 16 eight-GPU nodes, with SLOT=0..15." >&2
  exit 2
fi
slot="$1"
shift
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
# Use the Python environment activated on the evaluation node.
export PYTHONUNBUFFERED=1
exec "${PYTHON_BIN:-python}" scripts/launch_bright_repair.py run --workers 16 --slot "$slot" "$@"
