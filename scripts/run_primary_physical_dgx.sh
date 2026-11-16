#!/usr/bin/env bash
set -euo pipefail

BASE="/home/faculty/srmist14/MAJOR PROJECT"
REPO="$BASE/MAMAGuardNet"
SOURCE="${MAMAGUARD_SOURCE:-$REPO/post_review_staging/source_v9}"
PY="$REPO/.venv/bin/python"
RESULTS="$BASE/nnUNet_results"
PREPROCESSED="$BASE/nnUNet_preprocessed"
TARGET="$BASE/nnUNet_raw/Dataset501_MAMAMIA3Phase/labelsTr"
WORKERS="${MAMAGUARD_CPU_WORKERS:-8}"
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] && (( WORKERS <= $(nproc) ))
test -x "$PY"
test -f "$SOURCE/scripts/check_fold_gate.py"
test -f "$SOURCE/scripts/evaluate_predictions.py"
test -d "$TARGET"
cd "$SOURCE"
export PYTHONPATH="$SOURCE/src" CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

OUT=$(mktemp -d "$REPO/reports/generated/primary_physical_XXXXXXXX")
printf 'Output directory: %s\n' "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
configs=()
for name in a0_no_augmentation a1_nnunet_standard a2_independent a3_shared a4_kinetics; do
    configs+=(--config "$SOURCE/configs/experiments/$name.yaml")
done
for fold in {0..4}; do
    "$PY" scripts/check_fold_gate.py --results-root "$RESULTS" \
        --preprocessed-root "$PREPROCESSED" "${configs[@]}" --fold "$fold" \
        --seed 20261031 --output "$OUT/fold_$fold.json"
done
for fold in {0..4}; do
    "$PY" - "$OUT/fold_$fold.json" <<'PY' |
import json
import sys
from pathlib import Path

for run in json.loads(Path(sys.argv[1]).read_text())["experiments"]:
    print(run["experiment"] + "\t" + run["fold_root"])
PY
    while IFS=$'\t' read -r experiment fold_root; do
        nice -n 10 "$PY" scripts/evaluate_predictions.py \
            --predictions "$fold_root/validation" --targets "$TARGET" \
            --output-csv "$OUT/${experiment}_fold${fold}.csv" --workers "$WORKERS"
    done
done
printf 'All 25 physical evaluations completed. Outputs: %s\n' "$OUT"
