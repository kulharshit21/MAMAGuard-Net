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
MAX_RESTARTS="${MAMAGUARD_MAX_RESTARTS:-50}"
RETRY_SECONDS="${MAMAGUARD_RETRY_SECONDS:-120}"
PREFLIGHT_STABLE_SAMPLES="${MAMAGUARD_PREFLIGHT_STABLE_SAMPLES:-3}"
PREFLIGHT_INTERVAL_SECONDS="${MAMAGUARD_PREFLIGHT_INTERVAL_SECONDS:-30}"
PREFLIGHT_TIMEOUT_SECONDS="${MAMAGUARD_PREFLIGHT_TIMEOUT_SECONDS:-86400}"
PREFLIGHT_MIN_FREE_MIB="${MAMAGUARD_PREFLIGHT_MIN_FREE_MIB:-36000}"

: "${nnUNet_preprocessed:?Set nnUNet_preprocessed}"
: "${nnUNet_results:?Set nnUNet_results}"

if [[ ! "$PREFLIGHT_STABLE_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAMAGUARD_PREFLIGHT_STABLE_SAMPLES must be a positive integer" >&2
  exit 64
fi
if [[ ! "$PREFLIGHT_INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAMAGUARD_PREFLIGHT_INTERVAL_SECONDS must be a positive integer" >&2
  exit 64
fi
if [[ ! "$PREFLIGHT_TIMEOUT_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAMAGUARD_PREFLIGHT_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 64
fi
if [[ ! "$PREFLIGHT_MIN_FREE_MIB" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAMAGUARD_PREFLIGHT_MIN_FREE_MIB must be a positive integer" >&2
  exit 64
fi

EXPERIMENT="$(python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["experiment"])' "$CONFIG")"
RUN_ROOT="$nnUNet_results/$EXPERIMENT/seed_$SEED"
REGISTRY="runs/registry_gpu${GPU}.json"
LOG_DIR="reports/generated/training/$EXPERIMENT/fold_$FOLD"
mkdir -p "$LOG_DIR" "$(dirname "$REGISTRY")"

export MAMAGUARD_GPUS="$GPU"
export MAMAGUARD_SAVE_EVERY="${MAMAGUARD_SAVE_EVERY:-10}"
export nnUNet_compile="${nnUNet_compile:-false}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-12}"

attempt=0
while (( attempt <= MAX_RESTARTS )); do
  final_checkpoint="$(find "$RUN_ROOT" -type f \
    \( -name 'checkpoint_final.pth' -o -name 'checkpoint_final.safetensors' \) \
    -path "*/fold_${FOLD}/*" -print -quit 2>/dev/null || true)"
  if [[ -n "$final_checkpoint" ]]; then
    echo "Training already complete: $final_checkpoint"
    exit 0
  fi

  python -m mamaguardnet.cli gpu-preflight \
    --gpus "$GPU" --min-free-mib "$PREFLIGHT_MIN_FREE_MIB" --max-utilization 10 \
    --stable-samples "$PREFLIGHT_STABLE_SAMPLES" \
    --interval-seconds "$PREFLIGHT_INTERVAL_SECONDS" \
    --timeout-seconds "$PREFLIGHT_TIMEOUT_SECONDS" --wait \
    --output "$LOG_DIR/gpu_preflight_attempt_${attempt}.json"
  preflight_rc=$?
  if (( preflight_rc != 0 )); then
    echo "GPU preflight timed out; retrying in ${RETRY_SECONDS}s" >&2
    sleep "$RETRY_SECONDS"
    continue
  fi

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

  log="$LOG_DIR/attempt_${attempt}.log"
  python -m mamaguardnet.cli train "$CONFIG" \
    --registry "$REGISTRY" --execute --allow-duplicate \
    --fold "$FOLD" --seed "$SEED" "${resume_args[@]}" 2>&1 | tee "$log"
  train_rc=${PIPESTATUS[0]}
  if (( train_rc == 0 )); then
    echo "Training completed successfully on GPU $GPU, fold $FOLD"
    exit 0
  fi

  attempt=$((attempt + 1))
  echo "Training exited with code $train_rc; restart $attempt/$MAX_RESTARTS in ${RETRY_SECONDS}s" >&2
  sleep "$RETRY_SECONDS"
done

echo "Exceeded maximum restart count: $MAX_RESTARTS" >&2
exit 1
