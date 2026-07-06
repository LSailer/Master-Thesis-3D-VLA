#!/bin/bash
#SBATCH --job-name=r2d-graph-vs-buf
#SBATCH --partition=gpu_h100,gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=output/bench/r2d-graph-vs-buf-%j.out
#SBATCH --error=output/bench/r2d-graph-vs-buf-%j.err

set -euo pipefail

# Timing comparison on H100: the live buffer's per-step costs (jitted voxel
# add of a 518x518 frame at production capacity 2^23, 262k even-stride
# snapshot) vs the graph-structure candidate costs (jaxkd k=16 graph build,
# high-pass variation scores, Gumbel top-k sampling, 3x64 GCN forward) over
# the real saved 210k-point house cloud. Answers what adopting graph-based
# snapshots/encoding would add to the ~158 ms/step production loop.

mkdir -p output/bench

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Dirty: $(git status --porcelain | wc -l) files"

export PYTHONFAULTHANDLER=1

uv run python scripts/r2dreamer/bench_graph_vs_buffer.py --iters 30 --cuda

echo "=== Bench done ==="
