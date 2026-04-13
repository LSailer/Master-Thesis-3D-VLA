#!/bin/bash
#SBATCH --job-name=random-baseline-l1
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=output/baselines/random-l1/slurm-%j.out
#SBATCH --error=output/baselines/random-l1/slurm-%j.err

mkdir -p output/baselines/random-l1

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

uv run python modules/baselines/random_agent.py \
    --curriculum_path data/curriculum/level1_1house_1goal.json \
    --output_dir "output/baselines/random-l1" \
    --max_episode_steps 500 \
    --seed 42

echo "Done at $(date)"
