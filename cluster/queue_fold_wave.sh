#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 COMPLETED_FOLD NEXT_FOLD [SEED]" >&2
  exit 64
fi

COMPLETED_FOLD="$1"
NEXT_FOLD="$2"
SEED="${3:-20261031}"
POLL_SECONDS="${MAMAGUARD_GATE_POLL_SECONDS:-300}"

if [[ ! "$COMPLETED_FOLD" =~ ^[0-4]$ || ! "$NEXT_FOLD" =~ ^[0-4]$ ]]; then
  echo "Folds must be integers from 0 through 4" >&2
  exit 64
fi
if (( NEXT_FOLD != COMPLETED_FOLD + 1 )); then
  echo "NEXT_FOLD must immediately follow COMPLETED_FOLD" >&2
  exit 64
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_PARENT="$(dirname "$PROJECT_ROOT")"
cd "$PROJECT_ROOT"

export nnUNet_raw="${nnUNet_raw:-$PROJECT_PARENT/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$PROJECT_PARENT/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$PROJECT_PARENT/nnUNet_results}"

CONFIGS=(
  configs/experiments/a0_no_augmentation.yaml
  configs/experiments/a1_nnunet_standard.yaml
  configs/experiments/a2_independent.yaml
  configs/experiments/a3_shared.yaml
  configs/experiments/a4_kinetics.yaml
)
GPUS=(3 5 7 0 6)
REPORT="reports/generated/fold_${COMPLETED_FOLD}_completion_gate.json"
APPROVAL_DIR="runs/approvals"
APPROVAL="$APPROVAL_DIR/ALLOW_FOLD_${NEXT_FOLD}_SEED_${SEED}"
mkdir -p "$(dirname "$REPORT")" "$APPROVAL_DIR"

gate_command=(
  python scripts/check_fold_gate.py
  --results-root "$nnUNet_results"
  --preprocessed-root "$nnUNet_preprocessed"
  --fold "$COMPLETED_FOLD"
  --seed "$SEED"
  --output "$REPORT"
)
for config in "${CONFIGS[@]}"; do
  gate_command+=(--config "$config")
done

echo "Waiting for all five fold-${COMPLETED_FOLD} experiments to pass the completion gate."
until "${gate_command[@]}" >/dev/null; do
  echo "$(date --iso-8601=seconds): completion gate not ready; checking again in ${POLL_SECONDS}s"
  sleep "$POLL_SECONDS"
done

echo "Fold ${COMPLETED_FOLD} artifacts passed: $REPORT"
echo "Waiting for explicit approval file: $APPROVAL"
while [[ ! -f "$APPROVAL" ]]; do
  sleep "$POLL_SECONDS"
done

"${gate_command[@]}" >/dev/null

for index in "${!GPUS[@]}"; do
  session="mamaguard_g${GPUS[$index]}_f${NEXT_FOLD}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Refusing to replace existing tmux session: $session" >&2
    exit 1
  fi
done

for index in "${!GPUS[@]}"; do
  gpu="${GPUS[$index]}"
  config="${CONFIGS[$index]}"
  session="mamaguard_g${gpu}_f${NEXT_FOLD}"
  tmux new-session -d -s "$session" \
    "cd '$PROJECT_ROOT' && source .venv/bin/activate && \
     export nnUNet_raw='$nnUNet_raw' nnUNet_preprocessed='$nnUNet_preprocessed' \
            nnUNet_results='$nnUNet_results' && \
     bash scripts/resilient_train.sh '$config' '$gpu' '$NEXT_FOLD' '$SEED'"
  echo "Started $session: GPU $gpu, fold $NEXT_FOLD, config $config"
done

mv "$APPROVAL" "${APPROVAL}.consumed.$(date +%Y%m%dT%H%M%S)"
echo "Fold ${NEXT_FOLD} wave launched after a fresh completion-gate check."
