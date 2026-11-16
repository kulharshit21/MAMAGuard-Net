#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 3 ]]; then
  echo "Usage: $0 [COMPLETED_FOLD] [FINAL_FOLD] [SEED]" >&2
  exit 64
fi

COMPLETED_FOLD="${1:-0}"
FINAL_FOLD="${2:-4}"
SEED="${3:-20261031}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIGS=(
  configs/experiments/a0_no_augmentation.yaml
  configs/experiments/a1_nnunet_standard.yaml
  configs/experiments/a2_independent.yaml
  configs/experiments/a3_shared.yaml
  configs/experiments/a4_kinetics.yaml
)
GPUS=(3 5 7 0 6)

bash -n cluster/queue_experiment_folds.sh
mkdir -p reports/generated/queues

for gpu in "${GPUS[@]}"; do
  session="mamaguard_queue_g${gpu}"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "Refusing to replace existing queue controller: $session" >&2
    exit 1
  fi
done

for index in "${!GPUS[@]}"; do
  gpu="${GPUS[$index]}"
  config="${CONFIGS[$index]}"
  session="mamaguard_queue_g${gpu}"
  log="reports/generated/queues/gpu_${gpu}_folds_${COMPLETED_FOLD}_to_${FINAL_FOLD}.log"
  tmux new-session -d -s "$session" \
    "cd '$PROJECT_ROOT' && source .venv/bin/activate && \
     export MAMAGUARD_GATE_POLL_SECONDS=300 && \
     bash cluster/queue_experiment_folds.sh \
       '$config' '$gpu' '$COMPLETED_FOLD' '$FINAL_FOLD' '$SEED' \
       2>&1 | tee '$log'"
  echo "Started $session: config=$config gpu=$gpu folds=$COMPLETED_FOLD..$FINAL_FOLD"
done

echo "All per-experiment queue controllers started."
