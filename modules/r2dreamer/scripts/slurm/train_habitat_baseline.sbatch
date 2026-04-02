#!/bin/bash
#SBATCH --job-name=r2dreamer-habitat
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=output/r2dreamer-habitat-baseline/slurm-%j.out
#SBATCH --error=output/r2dreamer-habitat-baseline/slurm-%j.err

mkdir -p output/r2dreamer-habitat-baseline

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

uv run python modules/r2dreamer/scripts/run_jax_habitat.py \
    --steps 10000000 \
    --prefill 5000 \
    --output_dir "output/r2dreamer-habitat-baseline/run-${SLURM_JOB_ID}" \
    --seed "${SLURM_JOB_ID}" \
    --log_every 250 \
    --wandb_project 3d-vla-objectnav \
    --wandb_name "r2dreamer-habitat-10M-${SLURM_JOB_ID}"
