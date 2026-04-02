#!/bin/bash
#SBATCH --job-name=r2dreamer-habitat
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=output/r2dreamer-habitat-baseline/slurm-%j.out
#SBATCH --error=output/r2dreamer-habitat-baseline/slurm-%j.err

mkdir -p output/r2dreamer-habitat-baseline

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# 2.4M steps ≈ 23h at ~30 fps (measured in sanity check)
uv run python modules/r2dreamer/scripts/run_jax_habitat.py \
    --steps 2400000 \
    --prefill 5000 \
    --checkpoint_every 100000 \
    --output_dir "output/r2dreamer-habitat-baseline/run-${SLURM_JOB_ID}" \
    --seed "${SLURM_JOB_ID}" \
    --log_every 250 \
    --wandb_project 3d-vla-objectnav \
    --wandb_name "r2dreamer-habitat-2.4M-${SLURM_JOB_ID}"
