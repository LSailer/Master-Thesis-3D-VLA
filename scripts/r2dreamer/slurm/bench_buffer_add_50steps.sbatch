#!/bin/bash
#SBATCH --job-name=r2d-add50-bench
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=output/bench/r2d-add50-bench-%j.out
#SBATCH --error=output/bench/r2d-add50-bench-%j.err

set -euo pipefail

# 50-step timing of the fixed-shape HouseContextPoseBuffer.add on the real
# extract pipeline, plus a PLY snapshot of the accumulated house context.
# ~50 steps: model load+warmup (~1-2 min) + 50 * (~2s extract + add).

mkdir -p output/bench

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Dirty: $(git status --porcelain | wc -l) files"

export PYTHONFAULTHANDLER=1

uv run python scripts/r2dreamer/bench_buffer_add_50steps.py --steps 50

echo "=== Bench done ==="
