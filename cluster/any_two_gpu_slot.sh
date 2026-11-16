#!/usr/bin/env bash
set -u -o pipefail

if (( $# < 2 )); then
  echo "Usage: $0 SLOT_ID CONFIG [CONFIG ...]" >&2
  exit 64
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SLOT_ID="$1"
shift
CONFIGS=("$@")
SEED="${MAMAGUARD_SEED:-20261031}"
POLL_SECONDS="${MAMAGUARD_SHARED_POLL_SECONDS:-2}"
RETRY_SECONDS="${MAMAGUARD_RETRY_SECONDS:-10}"
MIN_FREE_MIB="${MAMAGUARD_SHARED_MIN_FREE_MIB:-26000}"
ORCH="$PROJECT_ROOT/reports/generated/orchestration"
SLOT_LOCK="$PROJECT_ROOT/runs/any_two_gpu_slot_${SLOT_ID}.lock"
DONE_FILE="$ORCH/any_two_gpu_slot_${SLOT_ID}.done"
ATTEMPT_PIDFILE="$ORCH/any_two_gpu_slot_${SLOT_ID}.attempt.pid"

if [[ ! "$SLOT_ID" =~ ^[01]$ ]]; then
  echo "SLOT_ID must be 0 or 1" >&2
  exit 64
fi
if [[ ! "$POLL_SECONDS" =~ ^[1-9][0-9]*$ || ! "$RETRY_SECONDS" =~ ^[1-9][0-9]*$ ||
      ! "$MIN_FREE_MIB" =~ ^[1-9][0-9]*$ ]]; then
  echo "Poll, retry and free-memory settings must be positive integers" >&2
  exit 64
fi

mkdir -p "$PROJECT_ROOT/runs" "$ORCH" "$PROJECT_ROOT/reports/generated/fold_gates"
cd "$PROJECT_ROOT"

exec 9>"$SLOT_LOCK"
if ! flock -n 9; then
  echo "Slot $SLOT_ID lock is already held" >&2
  exit 73
fi

SELECTED_GPU=""
GPU_LOCK_FD=""
ATTEMPT_PID=""

cleanup_attempt() {
  if [[ "$ATTEMPT_PID" =~ ^[0-9]+$ ]] && pgrep -g "$ATTEMPT_PID" >/dev/null 2>&1; then
    kill -TERM -- "-$ATTEMPT_PID" 2>/dev/null || true
    for _ in $(seq 1 10); do
      pgrep -g "$ATTEMPT_PID" >/dev/null 2>&1 || break
      sleep 1
    done
  fi
  rm -f "$ATTEMPT_PIDFILE"
  ATTEMPT_PID=""
}

release_gpu() {
  if [[ "$GPU_LOCK_FD" =~ ^[0-9]+$ ]]; then
    eval "exec ${GPU_LOCK_FD}>&-"
  fi
  GPU_LOCK_FD=""
  SELECTED_GPU=""
}

cleanup() {
  cleanup_attempt
  release_gpu
}
trap cleanup EXIT
trap 'cleanup; exit 143' TERM INT

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

gpu_uuid() {
  nvidia-smi -i "$1" --query-gpu=uuid --format=csv,noheader,nounits 2>/dev/null |
    awk 'NR == 1 {gsub(/^ +| +$/, "", $0); print $0}'
}

gpu_has_own_compute() {
  local gpu="$1" uuid pid owner
  uuid="$(gpu_uuid "$gpu")"
  while IFS=',' read -r process_uuid pid; do
    process_uuid="$(echo "$process_uuid" | xargs)"
    pid="$(echo "$pid" | tr -cd '0-9')"
    [[ "$process_uuid" == "$uuid" && "$pid" =~ ^[0-9]+$ ]] || continue
    owner="$(ps -o user= -p "$pid" 2>/dev/null | xargs || true)"
    [[ "$owner" == "$USER" ]] && return 0
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null)
  return 1
}

acquire_any_gpu() {
  local gpu free_mib lock_file fd
  while true; do
    for gpu in 0 1 2 3 4 5 6 7; do
      lock_file="$PROJECT_ROOT/runs/any_two_gpu_device_${gpu}.lock"
      exec {fd}>"$lock_file"
      if ! flock -n "$fd"; then
        eval "exec ${fd}>&-"
        continue
      fi

      if gpu_has_own_compute "$gpu"; then
        eval "exec ${fd}>&-"
        continue
      fi

      free_mib="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -cd '0-9')"
      if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
        SELECTED_GPU="$gpu"
        GPU_LOCK_FD="$fd"
        echo "$(date --iso-8601=seconds): slot=$SLOT_ID acquired gpu=$gpu free_mib=$free_mib"
        return 0
      fi
      eval "exec ${fd}>&-"
    done
    echo "$(date --iso-8601=seconds): slot=$SLOT_ID no eligible GPU; checking again in ${POLL_SECONDS}s"
    sleep "$POLL_SECONDS"
  done
}

run_attempt() {
  local config="$1" fold="$2" experiment registry run_root latest log rc
  experiment="$(experiment_name "$config")"
  registry="runs/registry_any_two_${experiment}.json"
  run_root="$nnUNet_results/$experiment/seed_$SEED"
  latest="$(find "$run_root" -type f \
    \( -name 'checkpoint_latest.pth' -o -name 'checkpoint_latest.safetensors' \) \
    -path "*/fold_${fold}/*" -print -quit 2>/dev/null || true)"
  log="reports/generated/training/$experiment/fold_$fold/any_two_attempt_$(date +%Y%m%dT%H%M%S).log"
  mkdir -p "$(dirname "$log")" "$(dirname "$registry")"

  acquire_any_gpu
  export MAMAGUARD_GPUS="$SELECTED_GPU"
  export MAMAGUARD_SAVE_EVERY="${MAMAGUARD_SAVE_EVERY:-10}"
  export nnUNet_compile="${nnUNet_compile:-false}"
  export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-6}"

  args=(python -m mamaguardnet.cli train "$config" --registry "$registry" --execute --allow-duplicate --fold "$fold" --seed "$SEED")
  [[ -n "$latest" ]] && args+=(--resume)

  echo "$(date --iso-8601=seconds): slot=$SLOT_ID gpu=$SELECTED_GPU experiment=$experiment fold=$fold resume_checkpoint=${latest:-none}"
  setsid "${args[@]}" >>"$log" 2>&1 &
  ATTEMPT_PID=$!
  echo "$ATTEMPT_PID" > "$ATTEMPT_PIDFILE"
  wait "$ATTEMPT_PID"
  rc=$?
  cleanup_attempt
  release_gpu
  return "$rc"
}

rm -f "$DONE_FILE"
echo "$(date --iso-8601=seconds): any-two slot started slot=$SLOT_ID configs=${CONFIGS[*]}"

for config in "${CONFIGS[@]}"; do
  [[ -f "$config" ]] || { echo "Missing config: $config" >&2; exit 66; }
  experiment="$(experiment_name "$config")"
  for fold in 3 4; do
    predecessor=$((fold - 1))
    until gate_ready "$config" "$predecessor"; do sleep "$POLL_SECONDS"; done
    gate_ready "$config" "$fold" && continue

    until gate_ready "$config" "$fold"; do
      if run_attempt "$config" "$fold"; then
        until gate_ready "$config" "$fold"; do sleep "$POLL_SECONDS"; done
      else
        rc=$?
        echo "$(date --iso-8601=seconds): slot=$SLOT_ID experiment=$experiment fold=$fold exit=$rc; reselecting any eligible GPU in ${RETRY_SECONDS}s" >&2
        sleep "$RETRY_SECONDS"
      fi
    done
    echo "$(date --iso-8601=seconds): slot=$SLOT_ID experiment=$experiment fold=$fold completion gate passed"
  done
done

touch "$DONE_FILE"
echo "$(date --iso-8601=seconds): any-two slot complete slot=$SLOT_ID"

