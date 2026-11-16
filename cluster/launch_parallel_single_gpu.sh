#!/usr/bin/env bash
set -euo pipefail

echo "Use only for independent experiments after each physical GPU passes preflight." >&2
echo "Preferred confirmatory runs use cluster/launch_ddp.sh across GPUs 1,5,6,7." >&2

GPU_IDS="${MAMAGUARD_GPUS:-1,5,6,7}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
if [[ $# -ne ${#GPU_ARRAY[@]} ]]; then
  echo "Provide exactly ${#GPU_ARRAY[@]} configs, one per GPU: $GPU_IDS" >&2
  exit 64
fi

CONFIGS=("$@")
for i in "${!GPU_ARRAY[@]}"; do
  gpu="${GPU_ARRAY[$i]}"
  config="${CONFIGS[$i]}"
  python -m mamaguardnet.cli gpu-preflight --gpus "$gpu" --min-free-mib 36000 --max-utilization 10
  MAMAGUARD_GPUS="$gpu" CUDA_VISIBLE_DEVICES="$gpu" python -m mamaguardnet.cli train "$config" --execute &
done
wait

