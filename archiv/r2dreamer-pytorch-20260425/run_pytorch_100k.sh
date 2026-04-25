#!/bin/bash
#SBATCH --job-name=pytorch-r2dreamer
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=output/slurm/pytorch-r2dreamer-%j.out
#SBATCH --error=output/slurm/pytorch-r2dreamer-%j.err

set -euo pipefail
mkdir -p output/slurm output/comparison

echo "=== PyTorch R2-Dreamer (standalone) on Crafter ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

uv run python modules/r2dreamer/scripts/run_pytorch_standalone.py \
    --steps 100000 \
    --prefill 5000 \
    --seed 0 \
    --rep_loss r2dreamer \
    --log_every 250 \
    --output output/comparison/pytorch_r2dreamer_metrics.csv

echo "Done at $(date)"
