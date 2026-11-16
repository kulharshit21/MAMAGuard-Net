#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 7 ]]; then
  echo "Usage: $0 CONFIG5 FOLD5 CONFIG6 FOLD6 CONFIG7 FOLD7 [SEED]" >&2
  exit 64
fi

SEED="${7:-20261031}"
GPUS=(5 6 7)
CONFIGS=("$1" "$3" "$5")
FOLDS=("$2" "$4" "$6")

: "${nnUNet_raw:?Set nnUNet_raw}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed}"
: "${nnUNet_results:?Set nnUNet_results}"

for index in 0 1 2; do
  gpu="${GPUS[$index]}"
  config="${CONFIGS[$index]}"
  fold="${FOLDS[$index]}"
  session="mamaguard_g${gpu}_f${fold}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Refusing to replace existing tmux session: $session" >&2
    exit 1
  fi
  tmux new-session -d -s "$session" \
    "cd '$PWD' && source .venv/bin/activate && \
     export nnUNet_raw='$nnUNet_raw' nnUNet_preprocessed='$nnUNet_preprocessed' \
            nnUNet_results='$nnUNet_results' && \
     bash scripts/resilient_train.sh '$config' '$gpu' '$fold' '$SEED'"
  echo "Started $session: GPU $gpu, fold $fold, config $config"
done
