#!/bin/bash
#SBATCH --job-name=profile-full-bf16-step
#SBATCH --partition=gpu_h100_il,gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=output/probes/step-time-opt/full-bf16-microbench/slurm-%j.out
#SBATCH --error=output/probes/step-time-opt/full-bf16-microbench/slurm-%j.err

# Direct worktree-local interpreter — deliberately NOT `uv run python`, which
# re-syncs (and here corrupts) the shared .venv. This microbench imports only
# the JAX stack, never habitat, so the broken habitat/numpy import is irrelevant.
set -euo pipefail
mkdir -p output/probes/step-time-opt/full-bf16-microbench

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
export JAX_PLATFORMS=""

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name --format=csv,noheader || true

.venv/bin/python -m scripts.profiling.profile_full_bf16_step \
    --batch 16 --seq 64 --iters 50 --warmup 5
