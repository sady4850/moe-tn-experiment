#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
.venv/bin/python -u scripts/projection_ablation.py > projection_ablation.log 2>&1
status=$?
printf '%s\n' "$status" > projection_ablation.exit
exit "$status"
