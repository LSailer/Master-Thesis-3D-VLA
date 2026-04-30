#!/bin/bash
#SBATCH --job-name=dreamerv3
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=output/dreamerv3/slurm-%j.out
#SBATCH --error=output/dreamerv3/slurm-%j.err

mkdir -p output/dreamerv3

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

uv run python -m src.dreamerv3.train \
    --total_steps 5000000 \
    --prefill_steps 5000 \
    --logdir "output/dreamerv3/run-${SLURM_JOB_ID}" \
    --seed "${SLURM_JOB_ID}"
