#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
.venv/bin/python -u scripts/calibration_crossover.py > calibration_crossover.log 2>&1
status=$?
printf '%s\n' "$status" > calibration_crossover.exit
exit "$status"
