#!/bin/bash
#SBATCH --job-name=r2d-L1-1house-chair
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=output/r2dreamer-curriculum-l1/slurm-%j.out
#SBATCH --error=output/r2dreamer-curriculum-l1/slurm-%j.err

mkdir -p output/r2dreamer-curriculum-l1

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Generate curriculum configs if not present
if [ ! -f data/curriculum/level1_1house_1goal.json ]; then
    echo "Generating curriculum configs..."
    uv run python scripts/environments/generate_curriculum.py
fi

# Level 1: 1 house (fK2vEV32Lag), chair only, 7499 train episodes
# 2.4M steps, single scene = focused learning
uv run python scripts/r2dreamer/run.py habitat-l1-cnn \
    --act_entropy 3e-2 \
    --steps 2400000 \
    --prefill 5000 \
    --checkpoint_every 100000 \
    --output_dir "output/r2dreamer-curriculum-l1/run-${SLURM_JOB_ID}" \
    --seed "${SLURM_JOB_ID}" \
    --log_every 250 \
    --wandb_project 3d-vla-objectnav \
    --wandb_name "r2d-L1-act-entropy-3e-2-${SLURM_JOB_ID}" \
    --wandb_tags "act-entropy-3e-2,baseline-restore,l1" \
    --val_data data/val_replay/val_200ep.npz \
    --val_loss_every 10000
