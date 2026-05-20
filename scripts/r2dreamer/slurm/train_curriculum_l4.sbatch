#!/bin/bash
#SBATCH --job-name=r2d-L4-buffix
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=output/r2dreamer-curriculum-l4/slurm-%j.out
#SBATCH --error=output/r2dreamer-curriculum-l4/slurm-%j.err

mkdir -p output/r2dreamer-curriculum-l4

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Generate curriculum configs if not present
if [ ! -f data/curriculum/level4_10houses_6goals.json ]; then
    echo "Generating curriculum configs..."
    uv run python scripts/environments/generate_curriculum.py
fi

# Level 4: 10 houses (4 easy, 4 medium, 2 hard) x 6 goal categories
# 2.4M steps, full curriculum — combines multi-scene (L3) + multi-goal (L2) generalization
uv run python scripts/r2dreamer/run_jax_habitat_l4.py \
    --steps 2400000 \
    --prefill 5000 \
    --checkpoint_every 100000 \
    --output_dir "output/r2dreamer-curriculum-l4/run-${SLURM_JOB_ID}" \
    --seed "${SLURM_JOB_ID}" \
    --log_every 250 \
    --wandb_project 3d-vla-objectnav \
    --wandb_name "r2d-L4-buffix-${SLURM_JOB_ID}" \
    --val_data data/val_replay/val_200ep.npz \
    --val_loss_every 10000
