#!/bin/bash
#SBATCH --job-name=dreamerv3-baseline
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=output/slurm/baseline-%j.out
#SBATCH --error=output/slurm/baseline-%j.err

# DreamerV3 baseline: 2D RGB on HM3D ObjectNav (JAX, 64x64)
# Submit: sbatch scripts/train_baseline.sh

set -euo pipefail

mkdir -p output/slurm

DATE=$(date +%Y%m%d)
LOGDIR="output/baseline-2d-jax-${DATE}"

echo "=== DreamerV3 Baseline Training ==="
echo "Date: ${DATE}"
echo "Logdir: ${LOGDIR}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "==================================="

uv run python -m src.dreamerv3.train \
    --obs_size 64 \
    --split train \
    --total_steps 10000000 \
    --logdir "${LOGDIR}" \
    --wandb_name "baseline-2d-jax-${DATE}" \
    --wandb_tags "baseline,full-run,jax,2d-rgb" \
    --wandb_group "baseline-runs" \
    --eval_every 50000 \
    --eval_episodes 10

echo "Training finished at $(date)"
