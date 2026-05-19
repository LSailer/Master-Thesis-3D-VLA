#!/bin/bash
#SBATCH --job-name=floorplan-fK2v
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=output/floorplan-fK2vEV32Lag/slurm-%j.out
#SBATCH --error=output/floorplan-fK2vEV32Lag/slurm-%j.err

mkdir -p output/floorplan-fK2vEV32Lag

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

PYTHONUNBUFFERED=1 uv run python scripts/environments/collect_floorplan_data.py

echo "Done at $(date)"
