#!/bin/bash
#SBATCH --job-name=r2d-house-branch-bench
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=output/bench/r2d-house-branch-bench-%j.out
#SBATCH --error=output/bench/r2d-house-branch-bench-%j.err

set -euo pipefail

# House-branch encoder cost vs snapshot size N (4k..8.4M) on H100.
# Decides static-N-masked vs bucketed split-boundary design.

mkdir -p output/bench

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

uv run python scripts/r2dreamer/bench_house_branch_sizes.py

echo "=== Bench done ==="
