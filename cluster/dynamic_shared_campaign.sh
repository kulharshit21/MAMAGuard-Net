#!/usr/bin/env bash
set -u -o pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_PARENT="$(dirname "$PROJECT_ROOT")"
SEED="${MAMAGUARD_SEED:-20261031}"
POLL_SECONDS="${MAMAGUARD_DYNAMIC_POLL_SECONDS:-2}"
MIN_FREE_MIB="${MAMAGUARD_SHARED_MIN_FREE_MIB:-25000}"
MAX_ACTIVE_LANES="${MAMAGUARD_MAX_ACTIVE_LANES:-5}"
RETRY_SECONDS="${MAMAGUARD_RETRY_SECONDS:-120}"
GPU_CANDIDATES_RAW="${MAMAGUARD_GPU_CANDIDATES:-0 1 2 3 4 5 6 7}"
LOCK_FILE="$PROJECT_ROOT/runs/dynamic_shared_campaign.lock"
SUPERVISOR_LOG="$PROJECT_ROOT/reports/generated/orchestration/dynamic_shared_campaign.log"

export nnUNet_raw="${nnUNet_raw:-$PROJECT_PARENT/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$PROJECT_PARENT/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$PROJECT_PARENT/nnUNet_results}"
export MAMAGUARD_SHARED_POLL_SECONDS="$POLL_SECONDS"
export MAMAGUARD_SHARED_MIN_FREE_MIB="$MIN_FREE_MIB"

mkdir -p "$PROJECT_ROOT/runs" "$PROJECT_ROOT/reports/generated/orchestration" \
  "$PROJECT_ROOT/reports/generated/fold_gates"
cd "$PROJECT_ROOT"

if [[ ! "$POLL_SECONDS" =~ ^[1-9][0-9]*$ || ! "$MIN_FREE_MIB" =~ ^[1-9][0-9]*$ ||
      ! "$MAX_ACTIVE_LANES" =~ ^[1-9][0-9]*$ || ! "$RETRY_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Dynamic scheduler settings must be positive integers" >&2
  exit 64
fi

read -r -a GPU_CANDIDATES <<< "$GPU_CANDIDATES_RAW"
if (( ${#GPU_CANDIDATES[@]} == 0 )); then
  echo "MAMAGUARD_GPU_CANDIDATES must contain at least one GPU index" >&2
  exit 64
fi
for gpu in "${GPU_CANDIDATES[@]}"; do
  if [[ ! "$gpu" =~ ^[0-7]$ ]]; then
    echo "Invalid GPU candidate: $gpu" >&2
    exit 64
  fi
done

experiment_name() {
  python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["experiment"])' "$1"
}

gate_ready() {
  local config="$1"
  local fold="$2"
  local experiment report
  experiment="$(experiment_name "$config")"
  report="reports/generated/fold_gates/$experiment/fold_${fold}_seed_${SEED}.json"
  mkdir -p "$(dirname "$report")"
  python scripts/check_fold_gate.py \
    --results-root "$nnUNet_results" \
    --preprocessed-root "$nnUNet_preprocessed" \
    --config "$config" --fold "$fold" --seed "$SEED" \
    --output "$report" >/dev/null 2>&1
}

run_lane() {
  local tag="$1"
  local config="$2"
  local gpu="$3"
  local experiment fold predecessor
  experiment="$(experiment_name "$config")"

  echo "$(date --iso-8601=seconds): lane started tag=$tag experiment=$experiment gpu=$gpu"
  for fold in 3 4; do
    predecessor=$((fold - 1))
    until gate_ready "$config" "$predecessor"; do
      echo "$(date --iso-8601=seconds): $experiment predecessor fold $predecessor not gated; waiting 3s"
      sleep 3
    done

    if gate_ready "$config" "$fold"; then
      echo "$(date --iso-8601=seconds): $experiment fold $fold already complete"
      continue
    fi

    echo "$(date --iso-8601=seconds): launching $experiment fold $fold on GPU $gpu"
    bash scripts/resilient_train_shared.sh "$config" "$gpu" "$fold" "$SEED" || exit $?

    until gate_ready "$config" "$fold"; do
      echo "$(date --iso-8601=seconds): $experiment fold $fold training returned; completion gate pending"
      sleep 3
    done
    echo "$(date --iso-8601=seconds): $experiment fold $fold completion gate passed"
  done

  echo "$(date --iso-8601=seconds): lane complete tag=$tag experiment=$experiment"
}

if [[ "${1:-}" == "lane" ]]; then
  if [[ $# -ne 4 ]]; then
    echo "Usage: $0 lane TAG CONFIG GPU" >&2
    exit 64
  fi
  run_lane "$2" "$3" "$4"
  exit $?
fi

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "Another dynamic shared campaign supervisor already holds $LOCK_FILE" >&2
  exit 73
fi

declare -a TAGS=(a0 a1 a2 a3 a4)
declare -A CONFIGS=(
  [a0]="configs/experiments/a0_no_augmentation.yaml"
  [a1]="configs/experiments/a1_nnunet_standard.yaml"
  [a2]="configs/experiments/a2_independent.yaml"
  [a3]="configs/experiments/a3_shared.yaml"
  [a4]="configs/experiments/a4_kinetics.yaml"
)

for tag in "${TAGS[@]}"; do
  if [[ ! -f "${CONFIGS[$tag]}" ]]; then
    echo "Missing experiment config: ${CONFIGS[$tag]}" >&2
    exit 66
  fi
done

lane_session() {
  local tag="$1"
  tmux list-sessions -F '#{session_name}' 2>/dev/null |
    grep -E "^mamaguard_dyn_${tag}_g[0-7]$" | head -n 1
}

lane_process_exists() {
  local config="$1"
  pgrep -u "$USER" -af 'python.*mamaguardnet\.cli train' 2>/dev/null |
    grep -F -- "$config" >/dev/null 2>&1
}

active_lane_count() {
  local tag config count=0 gpu_trainers=0
  for tag in "${TAGS[@]}"; do
    config="${CONFIGS[$tag]}"
    if [[ -n "$(lane_session "$tag")" ]] || lane_process_exists "$config"; then
      count=$((count + 1))
    fi
  done
  gpu_trainers="$(
    nvidia-smi --query-compute-apps=process_name --format=csv,noheader,nounits 2>/dev/null |
      grep -F "$PROJECT_ROOT/.venv/bin/python" |
      wc -l |
      awk '{$1=$1; print}'
  )"
  if [[ "$gpu_trainers" =~ ^[0-9]+$ ]] && (( gpu_trainers > count )); then
    count="$gpu_trainers"
  fi
  echo "$count"
}

gpu_reserved_by_lane() {
  local gpu="$1"
  tmux list-sessions -F '#{session_name}' 2>/dev/null |
    grep -Eq "^mamaguard_dyn_.*_g${gpu}$"
}

gpu_uuid() {
  nvidia-smi -i "$1" --query-gpu=uuid --format=csv,noheader,nounits |
    awk 'NR == 1 {gsub(/^ +| +$/, "", $0); print $0}'
}

gpu_free_mib() {
  nvidia-smi -i "$1" --query-gpu=memory.free --format=csv,noheader,nounits |
    awk 'NR == 1 {gsub(/[^0-9]/, "", $0); print $0}'
}

gpu_has_own_compute() {
  local gpu="$1"
  local uuid pid owner
  uuid="$(gpu_uuid "$gpu")"
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    owner="$(ps -o user= -p "$pid" 2>/dev/null | awk '{$1=$1; print}')"
    if [[ "$owner" == "$USER" ]]; then
      return 0
    fi
  done < <(
    nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null |
      awk -F',' -v wanted="$uuid" '
        {gsub(/^ +| +$/, "", $1); gsub(/[^0-9]/, "", $2)}
        $1 == wanted && $2 != "" {print $2}'
  )
  return 1
}

choose_gpu() {
  local gpu free_mib
  for gpu in "${GPU_CANDIDATES[@]}"; do
    gpu_reserved_by_lane "$gpu" && continue
    gpu_has_own_compute "$gpu" && continue
    free_mib="$(gpu_free_mib "$gpu")"
    if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= MIN_FREE_MIB )); then
      echo "$gpu"
      return 0
    fi
  done
  return 1
}

launch_lane() {
  local tag="$1"
  local config="$2"
  local gpu="$3"
  local session="mamaguard_dyn_${tag}_g${gpu}"
  local lane_log="$PROJECT_ROOT/reports/generated/orchestration/${session}.log"

  tmux new-session -d -s "$session" \
    "cd '$PROJECT_ROOT' && source .venv/bin/activate && \
     export nnUNet_raw='$nnUNet_raw' nnUNet_preprocessed='$nnUNet_preprocessed' \
            nnUNet_results='$nnUNet_results' MAMAGUARD_SEED='$SEED' \
            MAMAGUARD_DYNAMIC_POLL_SECONDS='$POLL_SECONDS' \
            MAMAGUARD_SHARED_POLL_SECONDS='$POLL_SECONDS' \
            MAMAGUARD_SHARED_MIN_FREE_MIB='$MIN_FREE_MIB' \
            MAMAGUARD_RETRY_SECONDS='$RETRY_SECONDS' && \
     bash cluster/dynamic_shared_campaign.sh lane '$tag' '$config' '$gpu' \
       2>&1 | tee -a '$lane_log'"
  echo "$(date --iso-8601=seconds): launched tag=$tag gpu=$gpu session=$session config=$config"
}

echo "$(date --iso-8601=seconds): dynamic shared supervisor started"
echo "poll_seconds=$POLL_SECONDS min_free_mib=$MIN_FREE_MIB max_active_lanes=$MAX_ACTIVE_LANES retry_seconds=$RETRY_SECONDS gpu_candidates=$GPU_CANDIDATES_RAW seed=$SEED"
echo "A GPU is rejected while it has any compute process owned by $USER."
echo "Other-user sharing is permitted only when at least ${MIN_FREE_MIB} MiB is free."

last_heartbeat=0
while true; do
  completed=0
  launched=0
  active_lanes="$(active_lane_count)"

  for tag in "${TAGS[@]}"; do
    config="${CONFIGS[$tag]}"
    if gate_ready "$config" 4; then
      completed=$((completed + 1))
      continue
    fi

    if [[ -n "$(lane_session "$tag")" ]]; then
      continue
    fi

    if lane_process_exists "$config"; then
      continue
    fi

    if (( active_lanes >= MAX_ACTIVE_LANES )); then
      continue
    fi

    gpu="$(choose_gpu || true)"
    if [[ -n "$gpu" ]]; then
      launch_lane "$tag" "$config" "$gpu"
      launched=$((launched + 1))
      active_lanes=$((active_lanes + 1))
    fi
  done

  if (( completed == ${#TAGS[@]} )); then
    echo "$(date --iso-8601=seconds): all five A0-A4 lanes passed fold-4 completion gates"
    exit 0
  fi

  now="$(date +%s)"
  if (( now - last_heartbeat >= 30 )); then
    echo "$(date --iso-8601=seconds): heartbeat completed_lanes=$completed active_sessions=$(tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -c '^mamaguard_dyn_' || true)"
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits || true
    last_heartbeat="$now"
  fi

  sleep "$POLL_SECONDS"
done
