#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 CONFIG.yaml [--resume|--evaluate-only]" >&2
  exit 64
fi

CONFIG_PATH="$1"
shift
GPU_IDS="${MAMAGUARD_GPUS:-1,5,6,7}"
export MAMAGUARD_GPUS="$GPU_IDS"
MIN_FREE_MIB="${MAMAGUARD_MIN_FREE_MIB:-36000}"
MAX_UTIL="${MAMAGUARD_MAX_GPU_UTIL:-10}"

python -m mamaguardnet.cli gpu-preflight \
  --gpus "$GPU_IDS" \
  --min-free-mib "$MIN_FREE_MIB" \
  --max-utilization "$MAX_UTIL" \
  --stable-samples "${MAMAGUARD_STABLE_SAMPLES:-3}" \
  --interval-seconds "${MAMAGUARD_POLL_SECONDS:-30}" \
  --timeout-seconds "${MAMAGUARD_WAIT_TIMEOUT_SECONDS:-86400}" \
  --wait --output reports/generated/gpu_preflight.json

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
exec python -m mamaguardnet.cli train "$CONFIG_PATH" --execute "$@"

