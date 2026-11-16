#!/usr/bin/env bash
set -u -o pipefail

RESULTS_ROOT="${nnUNet_results:-/home/faculty/srmist14/MAJOR PROJECT/nnUNet_results}"
MAX_LOG_AGE_SECONDS="${MAMAGUARD_STATUS_MAX_AGE:-900}"

EXPERIMENTS=(A0_no_augmentation A1_nnunet_standard A2_independent_phase_intensity A3_shared_affine A4_kinetics_constrained_primary)
GPUS=(3 5 7 0 6)

now=$(date +%s)
overall=0
echo "===== MAMAGUARD TRAINING STATUS ====="
date

echo
echo "===== TMUX ====="
tmux list-panes -a -F 'session=#{session_name} dead=#{pane_dead} pid=#{pane_pid} cmd=#{pane_current_command}' 2>/dev/null |
  awk '$1 ~ /^session=mamaguard_(g[03567]_f[0-4]|queue_g[03567]|backup)$/' |
  sort || true

echo
echo "===== EXPERIMENTS ====="
for index in "${!EXPERIMENTS[@]}"; do
  experiment="${EXPERIMENTS[$index]}"
  gpu="${GPUS[$index]}"
  root="$RESULTS_ROOT/$experiment"
  log=$(find "$root" -type f -name 'training_log_*.txt' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)
  echo "--- $experiment | GPU $gpu ---"
  if [[ -z "$log" ]]; then
    echo "status=MISSING_LOG"
    overall=1
    continue
  fi
  fold_dir=$(dirname "$log")
  fold=$(basename "$fold_dir" | sed -n 's/^fold_//p')
  final=$(find "$fold_dir" -maxdepth 1 -type f \( -name 'checkpoint_final.pth' -o -name 'checkpoint_final.safetensors' \) -print -quit 2>/dev/null || true)
  latest=$(find "$fold_dir" -maxdepth 1 -type f \( -name 'checkpoint_latest.pth' -o -name 'checkpoint_latest.safetensors' \) -print -quit 2>/dev/null || true)
  modified=$(stat -c %Y "$log")
  age=$((now - modified))
  completed=$(grep -c 'Epoch time:' "$log" || true)
  last_epoch=$(grep -E 'Epoch [0-9]+' "$log" | tail -n 1 || true)
  metrics=$(grep -E 'train_loss|val_loss|Pseudo dice' "$log" | tail -n 3 | tr '\n' ' ')
  if [[ -n "$final" ]]; then
    status=COMPLETED
  elif (( age <= MAX_LOG_AGE_SECONDS )); then
    status=ACTIVE
  else
    status=STALE_REVIEW
    overall=1
  fi
  echo "status=$status fold=${fold:-unknown} completed_epochs=$completed/1000 log_age_seconds=$age"
  echo "$last_epoch"
  echo "$metrics"
  [[ -n "$latest" ]] && stat -c 'latest_checkpoint=%y %s bytes %n' "$latest"
  [[ -n "$final" ]] && echo "final_checkpoint=$final"
  if tail -n 500 "$log" | grep -Eqi 'Traceback|CUDA out of memory|No space left|Killed'; then
    echo "fatal_marker=FOUND"
    overall=1
  else
    echo "fatal_marker=none"
  fi
done

echo
echo "===== GPU HEALTH ====="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free,temperature.gpu,power.draw --format=csv,noheader -i 0,3,5,6,7

echo
echo "===== STORAGE AND RAM ====="
df -h "/home/faculty/srmist14/MAJOR PROJECT" | tail -n 1
free -h | sed -n '1,2p'
exit "$overall"
