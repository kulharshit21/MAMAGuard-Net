#!/usr/bin/env bash
#SBATCH --job-name=mamaguard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=40
#SBATCH --mem=384G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
: "${MAMAGUARD_CONFIG:?Set MAMAGUARD_CONFIG to an experiment YAML path}"

# A scheduler allocates logical GPUs; do not force physical IDs inside Slurm.
python -m mamaguardnet.cli train "$MAMAGUARD_CONFIG" --execute

