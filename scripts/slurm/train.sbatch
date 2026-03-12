#!/bin/bash
#SBATCH --job-name=3dvla-train
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=output/slurm/%j.out
#SBATCH --error=output/slurm/%j.err

# Usage: sbatch --partition=gpu_h100 --time=04:00:00 scripts/slurm/train.sbatch

set -euo pipefail
mkdir -p output/slurm

# Wandb
export WANDB_PROJECT=3d-vla-objectnav
export WANDB_RUN_GROUP="${SLURM_JOB_NAME}"
export WANDB_TAGS="cluster,${SLURM_JOB_PARTITION:-unknown}"

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
uv run python -c "import jax; print(f'JAX devices: {jax.devices()}')"

# Training (placeholder — separate issue)
# uv run python src/train.py "$@"
echo "Training script not yet implemented — see issue."
