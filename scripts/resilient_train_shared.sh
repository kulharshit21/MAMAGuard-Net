#!/usr/bin/env bash
set -u -o pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 CONFIG GPU FOLD [SEED]" >&2
  exit 64
fi

CONFIG="$1"
GPU="$2"
FOLD="$3"
SEED="${4:-20261031}"
MAX_RESTARTS="${MAMAGUARD_MAX_RESTARTS:-200}"
RETRY_SECONDS="${MAMAGUARD_RETRY_SECONDS:-120}"
POLL_SECONDS="${MAMAGUARD_SHARED_POLL_SECONDS:-2}"
MIN_FREE_MIB="${MAMAGUARD_SHARED_MIN_FREE_MIB:-25000}"

: "${nnUNet_preprocessed:?Set nnUNet_preprocessed}"
: "${nnUNet_results:?Set nnUNet_results}"

if [[ ! "$GPU" =~ ^[0-7]$ || ! "$FOLD" =~ ^[0-4]$ ]]; then
  echo "GPU and fold must be integers in the range 0 through 7/4" >&2
  exit 64
fi
if [[ ! "$MAX_RESTARTS" =~ ^[1-9][0-9]*$ || ! "$RETRY_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Restart settings must be positive integers" >&2
  exit 64
fi
if [[ ! "$POLL_SECONDS" =~ ^[1-9][0-9]*$ || ! "$MIN_FREE_MIB" =~ ^[1-9][0-9]*$ ]]; then
  echo "Shared-GPU settings must be positive integers" >&2
  exit 64
fi

EXPERIMENT="$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["experiment"])' "$CONFIG")"
RUN_ROOT="$nnUNet_results/$EXPERIMENT/seed_$SEED"
REGISTRY="runs/registry_dynamic_${EXPERIMENT}.json"
LOG_DIR="reports/generated/training/$EXPERIMENT/fold_$FOLD"
mkdir -p "$LOG_DIR" "$(dirname "$REGISTRY")"

export MAMAGUARD_GPUS="$GPU"
export MAMAGUARD_SAVE_EVERY="${MAMAGUARD_SAVE_EVERY:-10}"
export nnUNet_compile="${nnUNet_compile:-false}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-12}"

free_memory_mib() {
  nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits |
    awk 'NR == 1 {gsub(/[^0-9]/, "", $0); print $0}'
}

wait_for_shared_capacity() {
  local free_mib
  while true; do
    free_mib="$(free_memory_mib)"
    if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
      echo "$(date --iso-8601=seconds): GPU $GPU has ${free_mib} MiB free; shared threshold is ${MIN_FREE_MIB} MiB"
      return 0
    fi
    echo "$(date --iso-8601=seconds): GPU $GPU has ${free_mib:-unknown} MiB free; waiting ${POLL_SECONDS}s"
    sleep "$POLL_SECONDS"
  done
}

attempt=0
while (( attempt <= MAX_RESTARTS )); do
  final_checkpoint="$(find "$RUN_ROOT" -type f \
    \( -name 'checkpoint_final.pth' -o -name 'checkpoint_final.safetensors' \) \
    -path "*/fold_${FOLD}/*" -print -quit 2>/dev/null || true)"
  if [[ -n "$final_checkpoint" ]]; then
    echo "Training already complete: $final_checkpoint"
    exit 0
  fi

  wait_for_shared_capacity

  resume_args=()
  latest_checkpoint="$(find "$RUN_ROOT" -type f \
    \( -name 'checkpoint_latest.pth' -o -name 'checkpoint_latest.safetensors' \) \
    -path "*/fold_${FOLD}/*" -print -quit 2>/dev/null || true)"
  if [[ -n "$latest_checkpoint" ]]; then
    echo "Resuming from: $latest_checkpoint"
    resume_args=(--resume)
  else
    echo "Starting fold $FOLD without an existing checkpoint"
  fi

  stamp="$(date +%Y%m%dT%H%M%S)"
  log="$LOG_DIR/shared_attempt_${attempt}_${stamp}.log"
  python -m mamaguardnet.cli train "$CONFIG" \
    --registry "$REGISTRY" --execute --allow-duplicate \
    --fold "$FOLD" --seed "$SEED" "${resume_args[@]}" 2>&1 | tee "$log"
  train_rc=${PIPESTATUS[0]}
  if (( train_rc == 0 )); then
    echo "Training completed successfully on shared GPU $GPU, fold $FOLD"
    exit 0
  fi

  attempt=$((attempt + 1))
  echo "Training exited with code $train_rc; restart $attempt/$MAX_RESTARTS in ${RETRY_SECONDS}s" >&2
  sleep "$RETRY_SECONDS"
done

echo "Exceeded maximum restart count: $MAX_RESTARTS" >&2
exit 1
