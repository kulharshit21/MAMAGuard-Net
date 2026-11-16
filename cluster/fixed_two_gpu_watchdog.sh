#!/usr/bin/env bash
set -u -o pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORCH="$PROJECT_ROOT/reports/generated/orchestration"
WATCHDOG_LOCK="$PROJECT_ROOT/runs/fixed_two_gpu_watchdog.lock"
POLL_SECONDS="${MAMAGUARD_WATCHDOG_SECONDS:-10}"

if [[ ! "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAMAGUARD_WATCHDOG_SECONDS must be a positive integer" >&2
  exit 64
fi

mkdir -p "$ORCH" "$PROJECT_ROOT/runs"
cd "$PROJECT_ROOT"
exec >>"$ORCH/fixed_two_gpu_watchdog.log" 2>&1

exec 9>"$WATCHDOG_LOCK"
if ! flock -n 9; then
  echo "Another fixed two-GPU watchdog already holds $WATCHDOG_LOCK" >&2
  exit 73
fi

slot_pidfile() { echo "$ORCH/fixed_two_gpu_gpu${1}.pid"; }
slot_done() { echo "$ORCH/fixed_two_gpu_gpu${1}.done"; }
slot_log() { echo "$ORCH/fixed_two_gpu_gpu${1}.log"; }

start_slot() {
  local gpu="$1"
  shift
  local pidfile pid
  pidfile="$(slot_pidfile "$gpu")"

  setsid bash cluster/fixed_two_gpu_slot.sh "$gpu" "$@" \
    >>"$(slot_log "$gpu")" 2>&1 < /dev/null &
  pid=$!
  echo "$pid" > "$pidfile"
  echo "$(date --iso-8601=seconds): started fixed slot gpu=$gpu pid=$pid configs=$*"
}

ensure_slot() {
  local gpu="$1"
  shift
  local pidfile done_file old_pid
  pidfile="$(slot_pidfile "$gpu")"
  done_file="$(slot_done "$gpu")"

  [[ -f "$done_file" ]] && return 0

  old_pid="$(cat "$pidfile" 2>/dev/null || true)"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    return 0
  fi

  if [[ "$old_pid" =~ ^[0-9]+$ ]] && pgrep -g "$old_pid" >/dev/null 2>&1; then
    echo "$(date --iso-8601=seconds): stale slot leader gpu=$gpu pid=$old_pid has surviving process group; sending TERM"
    kill -TERM -- "-$old_pid" 2>/dev/null || true
    sleep "$POLL_SECONDS"
    pgrep -g "$old_pid" >/dev/null 2>&1 && return 0
  fi

  start_slot "$gpu" "$@"
}

echo "$(date --iso-8601=seconds): fixed two-GPU watchdog started poll_seconds=$POLL_SECONDS allowed_gpus=5,6"

while true; do
  ensure_slot 5 \
    configs/experiments/a0_no_augmentation.yaml \
    configs/experiments/a4_kinetics.yaml

  ensure_slot 6 \
    configs/experiments/a2_independent.yaml \
    configs/experiments/a3_shared.yaml \
    configs/experiments/a1_nnunet_standard.yaml

  sleep "$POLL_SECONDS"
done

