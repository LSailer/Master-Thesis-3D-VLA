#!/bin/bash
#SBATCH --job-name=dreamer-easy
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=output/dreamerv3-easy/slurm-%j.out
#SBATCH --error=output/dreamerv3-easy/slurm-%j.err

mkdir -p output/dreamerv3-easy

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

uv run python -m src.dreamerv3.train \
    --total_steps 5000000 \
    --max_episode_steps 200 \
    --obs_size 64 \
    --prefill_steps 5000 \
    --save_every 500000 \
    --log_every 500 \
    --max_geodesic 5.0 \
    --split train \
    --logdir "output/dreamerv3-easy/run-${SLURM_JOB_ID}" \
    --seed "${SLURM_JOB_ID}"
