#!/usr/bin/env bash
set -u -o pipefail

if (( $# < 2 )); then
  echo "Usage: $0 GPU CONFIG [CONFIG ...]" >&2
  exit 64
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="$1"
shift
CONFIGS=("$@")
SEED="${MAMAGUARD_SEED:-20261031}"
POLL_SECONDS="${MAMAGUARD_SHARED_POLL_SECONDS:-2}"
RETRY_SECONDS="${MAMAGUARD_RETRY_SECONDS:-10}"
LOCK_FILE="$PROJECT_ROOT/runs/fixed_two_gpu_${GPU}.lock"
DONE_FILE="$PROJECT_ROOT/reports/generated/orchestration/fixed_two_gpu_gpu${GPU}.done"

if [[ "$GPU" != "5" && "$GPU" != "6" ]]; then
  echo "Fixed two-GPU slots permit only GPU 5 or GPU 6" >&2
  exit 64
fi

mkdir -p "$PROJECT_ROOT/runs" "$PROJECT_ROOT/reports/generated/orchestration" \
  "$PROJECT_ROOT/reports/generated/fold_gates"
cd "$PROJECT_ROOT"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "GPU $GPU slot lock is already held" >&2
  exit 73
fi

rm -f "$DONE_FILE"

experiment_name() {
  python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["experiment"])' "$1"
}

gate_ready() {
  local config="$1" fold="$2" experiment report
  experiment="$(experiment_name "$config")"
  report="reports/generated/fold_gates/$experiment/fold_${fold}_seed_${SEED}.json"
  mkdir -p "$(dirname "$report")"
  python scripts/check_fold_gate.py \
    --results-root "$nnUNet_results" \
    --preprocessed-root "$nnUNet_preprocessed" \
    --config "$config" --fold "$fold" --seed "$SEED" \
    --output "$report" >/dev/null 2>&1
}

echo "$(date --iso-8601=seconds): fixed slot started gpu=$GPU configs=${CONFIGS[*]}"

for config in "${CONFIGS[@]}"; do
  if [[ ! -f "$config" ]]; then
    echo "Missing experiment config: $config" >&2
    exit 66
  fi

  experiment="$(experiment_name "$config")"
  for fold in 3 4; do
    predecessor=$((fold - 1))
    until gate_ready "$config" "$predecessor"; do
      echo "$(date --iso-8601=seconds): gpu=$GPU experiment=$experiment predecessor_fold=$predecessor pending; checking in ${POLL_SECONDS}s"
      sleep "$POLL_SECONDS"
    done

    if gate_ready "$config" "$fold"; then
      echo "$(date --iso-8601=seconds): gpu=$GPU experiment=$experiment fold=$fold already complete"
      continue
    fi

    while ! gate_ready "$config" "$fold"; do
      echo "$(date --iso-8601=seconds): gpu=$GPU launching experiment=$experiment fold=$fold"
      if bash scripts/resilient_train_shared.sh "$config" "$GPU" "$fold" "$SEED"; then
        until gate_ready "$config" "$fold"; do
          echo "$(date --iso-8601=seconds): gpu=$GPU experiment=$experiment fold=$fold gate pending; checking in ${POLL_SECONDS}s"
          sleep "$POLL_SECONDS"
        done
      else
        rc=$?
        echo "$(date --iso-8601=seconds): gpu=$GPU experiment=$experiment fold=$fold controller_exit=$rc; retrying in ${RETRY_SECONDS}s" >&2
        sleep "$RETRY_SECONDS"
      fi
    done
    echo "$(date --iso-8601=seconds): gpu=$GPU experiment=$experiment fold=$fold completion gate passed"
  done
done

touch "$DONE_FILE"
echo "$(date --iso-8601=seconds): fixed slot complete gpu=$GPU"

