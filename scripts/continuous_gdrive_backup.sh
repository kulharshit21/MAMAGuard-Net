#!/usr/bin/env bash
set -u -o pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_PARENT="$(dirname "$PROJECT_ROOT")"
RESULTS_ROOT="${nnUNet_results:-$PROJECT_PARENT/nnUNet_results}"
REMOTE_ROOT="${MAMAGUARD_BACKUP_REMOTE:-gdrive:MAMAGuard-Net_Continuous_Backup_2026-11-01}"
INTERVAL_SECONDS="${MAMAGUARD_BACKUP_INTERVAL_SECONDS:-600}"
STABLE_AGE="${MAMAGUARD_BACKUP_STABLE_AGE:-5m}"
RUN_ONCE="${MAMAGUARD_BACKUP_ONCE:-0}"
RCLONE_BIN="${RCLONE_BIN:-$(command -v rclone 2>/dev/null || true)}"
STATE_DIR="$PROJECT_ROOT/runs/backup"
STATUS_FILE="$STATE_DIR/status.txt"
LOCK_FILE="$STATE_DIR/continuous_gdrive_backup.lock"

if [[ -z "$RCLONE_BIN" || ! -x "$RCLONE_BIN" ]]; then
  echo "rclone executable was not found" >&2
  exit 69
fi
if [[ ! "$INTERVAL_SECONDS" =~ ^[0-9]+$ ]] || (( INTERVAL_SECONDS < 300 )); then
  echo "MAMAGUARD_BACKUP_INTERVAL_SECONDS must be an integer of at least 300" >&2
  exit 64
fi
if [[ "$RUN_ONCE" != "0" && "$RUN_ONCE" != "1" ]]; then
  echo "MAMAGUARD_BACKUP_ONCE must be 0 or 1" >&2
  exit 64
fi
if [[ ! -d "$RESULTS_ROOT" ]]; then
  echo "nnU-Net results directory does not exist: $RESULTS_ROOT" >&2
  exit 66
fi

mkdir -p "$STATE_DIR"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another continuous backup process already holds $LOCK_FILE" >&2
  exit 73
fi

COMMON_ARGS=(
  --min-age "$STABLE_AGE"
  --transfers 2
  --checkers 8
  --retries 10
  --low-level-retries 20
  --retries-sleep 15s
  --stats 60s
  --stats-one-line
  --filter '- **/*.tmp'
  --filter '- **/*.partial'
  --filter '- **/*.part'
  --filter '- **/.nfs*'
  --filter '- **/__pycache__/**'
  --filter '- **/.pytest_cache/**'
  --filter '- **/.ruff_cache/**'
  --filter '- **/.mypy_cache/**'
  --filter '- **/*.pyc'
  --filter '- **/*.pyo'
)

copy_tree() {
  local source="$1"
  local destination="$2"
  shift 2
  [[ -d "$source" ]] || return 0
  "$RCLONE_BIN" copy "$source" "$destination" "${COMMON_ARGS[@]}" "$@"
}

write_status() {
  local state="$1"
  local temporary="${STATUS_FILE}.tmp"
  {
    echo "state=$state"
    echo "updated_at=$(date --iso-8601=seconds)"
    echo "remote=$REMOTE_ROOT"
    echo "results_root=$RESULTS_ROOT"
    echo "stable_age=$STABLE_AGE"
    echo "interval_seconds=$INTERVAL_SECONDS"
  } >"$temporary"
  mv "$temporary" "$STATUS_FILE"
}

echo "Continuous MAMAGuard backup started."
echo "Project: $PROJECT_ROOT"
echo "Results: $RESULTS_ROOT"
echo "Remote: $REMOTE_ROOT"
echo "Files newer than $STABLE_AGE are deferred until a later cycle."
echo "Raw datasets, nnUNet_raw, nnUNet_preprocessed, and environments are excluded."

while true; do
  cycle_failed=0
  cycle_started="$(date --iso-8601=seconds)"
  echo "$cycle_started: backup cycle started"
  write_status running

  # Preserve current/best/final checkpoints first. Files still being written are
  # deferred by --min-age and will be captured in a later cycle.
  copy_tree "$RESULTS_ROOT" "$REMOTE_ROOT/nnUNet_results" \
    --filter '+ **/checkpoint_best.*' \
    --filter '+ **/checkpoint_final.*' \
    --filter '+ **/checkpoint_latest.*' \
    --filter '- **' || cycle_failed=1

  # Copy lightweight reproducibility material before potentially large
  # validation volumes and probability archives.
  for relative in configs src scripts cluster tests manifests reports runs docs; do
    copy_tree "$PROJECT_ROOT/$relative" "$REMOTE_ROOT/repository/$relative" || cycle_failed=1
  done

  "$RCLONE_BIN" copy "$PROJECT_ROOT" "$REMOTE_ROOT/repository" \
    "${COMMON_ARGS[@]}" \
    --max-depth 1 \
    --filter '+ /README*' \
    --filter '+ /pyproject.toml' \
    --filter '+ /requirements*.txt' \
    --filter '+ /*.md' \
    --filter '+ /*.yaml' \
    --filter '+ /*.yml' \
    --filter '+ /*.json' \
    --filter '- **' || cycle_failed=1

  # Copy all remaining result artifacts: native logs, plans, validation
  # predictions, summaries, probability archives, metrics, and figures.
  copy_tree "$RESULTS_ROOT" "$REMOTE_ROOT/nnUNet_results" \
    --filter '- **/checkpoint_best.*' \
    --filter '- **/checkpoint_final.*' \
    --filter '- **/checkpoint_latest.*' || cycle_failed=1

  if (( cycle_failed == 0 )); then
    write_status success
    echo "$(date --iso-8601=seconds): backup cycle completed successfully"
  else
    write_status retry_required
    echo "$(date --iso-8601=seconds): backup cycle had errors; the next cycle will retry" >&2
  fi

  if [[ "$RUN_ONCE" == "1" ]]; then
    exit "$cycle_failed"
  fi

  sleep "$INTERVAL_SECONDS"
done
