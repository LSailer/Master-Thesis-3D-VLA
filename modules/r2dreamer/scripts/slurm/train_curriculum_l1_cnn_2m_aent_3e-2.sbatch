#!/bin/bash
#SBATCH --job-name=cnn_2m_aent3e-2
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=output/r2dreamer-curriculum-l1/slurm-%j.out
#SBATCH --error=output/r2dreamer-curriculum-l1/slurm-%j.err

set -euo pipefail

mkdir -p output/r2dreamer-curriculum-l1

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Run: CNN, L1, 2M steps, act_entropy=3e-2 (baseline), issue #142 actor metrics"
echo "Git: $(git rev-parse --short HEAD) branch $(git branch --show-current) dirty=$(test -n "$(git status --short)" && echo yes || echo no)"

if [ ! -f data/curriculum/level1_1house_1goal.json ]; then
    echo "Generating curriculum configs..."
    uv run python modules/envs/scripts/generate_curriculum.py
fi

uv run python modules/r2dreamer/scripts/run_jax_habitat.py \
    --act_entropy 3e-2 \
    --steps 2000000 \
    --prefill 5000 \
    --checkpoint_every 100000 \
    --output_dir "output/r2dreamer-curriculum-l1/run-${SLURM_JOB_ID}" \
    --seed "${SLURM_JOB_ID}" \
    --log_every 250 \
    --wandb_project 3d-vla-objectnav \
    --wandb_name "cnn_l1_2m_aent3e-2-${SLURM_JOB_ID}" \
    --wandb_tags "cnn,2m,48h,act_entropy_3e-2,baseline-entropy,issue-142,actor-return-stats,ret-scale-diag,advantage-diag,weight-mean-diag" \
    --val_data data/val_replay/val_200ep.npz \
    --val_loss_every 10000
