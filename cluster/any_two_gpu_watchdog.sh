#!/usr/bin/env bash
set -u -o pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORCH="$PROJECT_ROOT/reports/generated/orchestration"
LOCK="$PROJECT_ROOT/runs/any_two_gpu_watchdog.lock"
POLL_SECONDS="${MAMAGUARD_WATCHDOG_SECONDS:-10}"

mkdir -p "$ORCH" "$PROJECT_ROOT/runs"
cd "$PROJECT_ROOT"
exec >>"$ORCH/any_two_gpu_watchdog.log" 2>&1
exec 9>"$LOCK"
flock -n 9 || { echo "Any-two watchdog lock already held" >&2; exit 73; }

slot_pidfile() { echo "$ORCH/any_two_gpu_slot_${1}.pid"; }
attempt_pidfile() { echo "$ORCH/any_two_gpu_slot_${1}.attempt.pid"; }
done_file() { echo "$ORCH/any_two_gpu_slot_${1}.done"; }
slot_log() { echo "$ORCH/any_two_gpu_slot_${1}.log"; }

cleanup_group() {
  local pid="$1"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 0
  if pgrep -g "$pid" >/dev/null 2>&1; then
    kill -TERM -- "-$pid" 2>/dev/null || true
    for _ in $(seq 1 10); do
      pgrep -g "$pid" >/dev/null 2>&1 || break
      sleep 1
    done
  fi
}

start_slot() {
  local slot="$1"
  shift
  local pid
  setsid bash cluster/any_two_gpu_slot.sh "$slot" "$@" >>"$(slot_log "$slot")" 2>&1 < /dev/null &
  pid=$!
  echo "$pid" > "$(slot_pidfile "$slot")"
  echo "$(date --iso-8601=seconds): started slot=$slot pid=$pid configs=$*"
}

ensure_slot() {
  local slot="$1"
  shift
  local pid attempt
  [[ -f "$(done_file "$slot")" ]] && return 0
  pid="$(cat "$(slot_pidfile "$slot")" 2>/dev/null || true)"
  [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && return 0

  attempt="$(cat "$(attempt_pidfile "$slot")" 2>/dev/null || true)"
  cleanup_group "$attempt"
  cleanup_group "$pid"
  rm -f "$(attempt_pidfile "$slot")" "$(slot_pidfile "$slot")"
  start_slot "$slot" "$@"
}

echo "$(date --iso-8601=seconds): any-two watchdog started max_slots=2 poll_seconds=$POLL_SECONDS"
while true; do
  ensure_slot 0 configs/experiments/a0_no_augmentation.yaml configs/experiments/a4_kinetics.yaml
  ensure_slot 1 configs/experiments/a2_independent.yaml configs/experiments/a3_shared.yaml configs/experiments/a1_nnunet_standard.yaml
  sleep "$POLL_SECONDS"
done

