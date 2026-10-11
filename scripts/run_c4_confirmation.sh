#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
.venv/bin/python -u scripts/c4_layer3_confirmation.py > c4_confirmation.log 2>&1
status=$?
printf '%s\n' "$status" > c4_confirmation.exit
exit "$status"
