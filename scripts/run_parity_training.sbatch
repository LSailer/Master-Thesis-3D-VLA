#!/bin/bash
#SBATCH --job-name=r2d-parity
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=output/parity/slurm_%j.log
#SBATCH --error=output/parity/slurm_%j.log

set -euo pipefail

PROJECT_DIR="/pfs/data6/home/ul/ul_student/ul_hfj15/Master-Thesis-3D-VLA"
cd "$PROJECT_DIR"

echo "=== R2-Dreamer Parity Training ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"
echo ""

uv run python modules/r2dreamer/scripts/run_parity_training.py --train-steps 100000 --jax-only

echo ""
echo "=== Done at $(date) ==="
