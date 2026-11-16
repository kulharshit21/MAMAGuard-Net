#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "Usage: $0 CONFIG GPU COMPLETED_FOLD [FINAL_FOLD] [SEED]" >&2
  exit 64
fi

CONFIG="$1"
GPU="$2"
COMPLETED_FOLD="$3"
FINAL_FOLD="${4:-4}"
SEED="${5:-20261031}"
POLL_SECONDS="${MAMAGUARD_GATE_POLL_SECONDS:-300}"
PREFLIGHT_STABLE_SAMPLES="${MAMAGUARD_PREFLIGHT_STABLE_SAMPLES:-3}"
PREFLIGHT_INTERVAL_SECONDS="${MAMAGUARD_PREFLIGHT_INTERVAL_SECONDS:-30}"

if [[ ! "$GPU" =~ ^[0-7]$ ]]; then
  echo "GPU must be an integer from 0 through 7" >&2
  exit 64
fi
if [[ ! "$COMPLETED_FOLD" =~ ^[0-4]$ || ! "$FINAL_FOLD" =~ ^[0-4]$ ]]; then
  echo "Folds must be integers from 0 through 4" >&2
  exit 64
fi
if (( FINAL_FOLD < COMPLETED_FOLD )); then
  echo "FINAL_FOLD must be greater than or equal to COMPLETED_FOLD" >&2
  exit 64
fi
if (( POLL_SECONDS < 3 )); then
  echo "MAMAGUARD_GATE_POLL_SECONDS must be at least 3" >&2
  exit 64
fi
if [[ ! "$PREFLIGHT_STABLE_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAMAGUARD_PREFLIGHT_STABLE_SAMPLES must be a positive integer" >&2
  exit 64
fi
if [[ ! "$PREFLIGHT_INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAMAGUARD_PREFLIGHT_INTERVAL_SECONDS must be a positive integer" >&2
  exit 64
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_PARENT="$(dirname "$PROJECT_ROOT")"
cd "$PROJECT_ROOT"

if [[ ! -f "$CONFIG" ]]; then
  echo "Experiment config does not exist: $CONFIG" >&2
  exit 66
fi

export nnUNet_raw="${nnUNet_raw:-$PROJECT_PARENT/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$PROJECT_PARENT/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$PROJECT_PARENT/nnUNet_results}"

EXPERIMENT="$(python -c \
  'import sys, yaml; print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["experiment"])' \
  "$CONFIG")"
REPORT_DIR="reports/generated/fold_gates/$EXPERIMENT"
mkdir -p "$REPORT_DIR"

gate_ready() {
  local fold="$1"
  local report="$REPORT_DIR/fold_${fold}_seed_${SEED}.json"
  python scripts/check_fold_gate.py \
    --results-root "$nnUNet_results" \
    --preprocessed-root "$nnUNet_preprocessed" \
    --config "$CONFIG" \
    --fold "$fold" \
    --seed "$SEED" \
    --output "$report" >/dev/null
}

echo "Queue controller: experiment=$EXPERIMENT gpu=$GPU seed=$SEED"
echo "Fold range: completed fold $COMPLETED_FOLD through final fold $FINAL_FOLD"
echo "The external Yunnan cohort is not referenced by this controller."

current_fold="$COMPLETED_FOLD"
while true; do
  echo "Waiting for $EXPERIMENT fold $current_fold to pass its completion gate."
  until gate_ready "$current_fold"; do
    echo "$(date --iso-8601=seconds): $EXPERIMENT fold $current_fold not ready; checking again in ${POLL_SECONDS}s"
    sleep "$POLL_SECONDS"
  done
  echo "$EXPERIMENT fold $current_fold passed its completion gate."

  if (( current_fold == FINAL_FOLD )); then
    echo "$EXPERIMENT completed every scheduled fold through fold $FINAL_FOLD."
    exit 0
  fi

  next_fold=$((current_fold + 1))
  if gate_ready "$next_fold"; then
    echo "$EXPERIMENT fold $next_fold is already complete; no launch required."
    current_fold="$next_fold"
    continue
  fi

  session="mamaguard_g${GPU}_f${next_fold}"
  if tmux has-session -t "$session" 2>/dev/null; then
    pane_command="$(tmux list-panes -t "$session" -F '#{pane_start_command}')"
    if [[ "$pane_command" != *"$CONFIG"* ]]; then
      echo "Session $session exists but does not reference $CONFIG; refusing to assume ownership." >&2
      exit 1
    fi
    echo "Session $session already exists for $EXPERIMENT; waiting on its completion gate."
  else
    tmux new-session -d -s "$session" \
      "cd '$PROJECT_ROOT' && source .venv/bin/activate && \
       export nnUNet_raw='$nnUNet_raw' nnUNet_preprocessed='$nnUNet_preprocessed' \
              nnUNet_results='$nnUNet_results' \
              MAMAGUARD_PREFLIGHT_STABLE_SAMPLES='$PREFLIGHT_STABLE_SAMPLES' \
              MAMAGUARD_PREFLIGHT_INTERVAL_SECONDS='$PREFLIGHT_INTERVAL_SECONDS' && \
       bash scripts/resilient_train.sh '$CONFIG' '$GPU' '$next_fold' '$SEED'"
    echo "Started $session: experiment=$EXPERIMENT gpu=$GPU fold=$next_fold seed=$SEED"
  fi

  current_fold="$next_fold"
done
