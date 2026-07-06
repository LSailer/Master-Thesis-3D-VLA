#!/bin/bash
#SBATCH --job-name=r2d-align-check
#SBATCH --partition=gpu_h100,gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=output/check/r2d-align-check-%j.out
#SBATCH --error=output/check/r2d-align-check-%j.err

set -euo pipefail

# Diagnoses whether the persistent per-scene house buffer accumulates
# MISALIGNED copies of the house across episodes (VGGT world frame is
# re-anchored at each extractor.reset()). Motivated by smoke 5706115:
# 3.38M points in 1200 steps, big jump across the first episode boundary.
# See scripts/r2dreamer/check_episode_frame_alignment.py for metrics/verdict.

mkdir -p output/check

echo "Job $SLURM_JOB_ID on $(hostname) ($SLURM_JOB_PARTITION) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Commit: $(git rev-parse --short HEAD), dirty: $(git status --porcelain | wc -l) files"

if [ ! -f data/curriculum/level1_1house_1goal.json ]; then
    uv run python scripts/environments/generate_curriculum.py
fi

uv run python scripts/r2dreamer/check_episode_frame_alignment.py \
    --episodes 3 \
    --steps 60 \
    --seed 42 \
    --output "output/check/episode_frame_alignment_${SLURM_JOB_ID}"
