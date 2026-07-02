#!/bin/bash
#SBATCH --job-name=r2d-ext-add-bench
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=output/bench/r2d-ext-add-bench-%j.out
#SBATCH --error=output/bench/r2d-ext-add-bench-%j.err

set -euo pipefail

# Microbenchmark: VGGT extract vs HouseContextPoseBuffer.add per step, on the
# real GPU extractor with the habitat-l1-vggt-house-points-pose config.
# Reports extract_ms vs add_full_ms vs add_stride8_ms so we can attribute the
# per-step wall clock between GPU forward and the host-side voxel loop.
# ~30 steps: model load+warmup (~1-2 min) + 30*(~2s extract + ~0.5s full add).

mkdir -p output/bench

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Dirty: $(git status --porcelain | wc -l) files"

export PYTHONFAULTHANDLER=1

uv run python scripts/r2dreamer/bench_extract_vs_add.py

echo "=== Bench done ==="
