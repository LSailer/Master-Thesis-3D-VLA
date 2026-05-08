#!/bin/bash
#SBATCH --job-name=r2d-ab-internal-crafter
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=output/runs/r2dreamer-parity-crafter/internal-%j.out
#SBATCH --error=output/runs/r2dreamer-parity-crafter/internal-%j.err

mkdir -p output/runs/r2dreamer-parity-crafter

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# A/B with external/r2dreamer (PyTorch ref): both write to project r2dreamer-parity-crafter.
# Internal impl: JAX, CNN encoder (only Crafter-compatible option in modules/r2dreamer).
# Step budget 1.0M to match external default 1.01e6.

uv run python modules/r2dreamer/scripts/run_jax_crafter.py \
    --steps 1000000 \
    --prefill 5000 \
    --seed 0 \
    --log_every 250 \
    --checkpoint_every 100000 \
    --output_dir "output/runs/r2dreamer-parity-crafter/internal-${SLURM_JOB_ID}" \
    --wandb_project "r2dreamer-parity-crafter" \
    --wandb_name "internal-jax-crafter-cnn-s0-${SLURM_JOB_ID}" \
    --wandb_tags "ab,parity,internal,jax,crafter,cnn"
