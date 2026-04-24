#!/bin/bash
#SBATCH --job-name=r2d-L2-buffix
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=output/r2dreamer-curriculum-l2/slurm-%j.out
#SBATCH --error=output/r2dreamer-curriculum-l2/slurm-%j.err

mkdir -p output/r2dreamer-curriculum-l2

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Generate curriculum configs if not present
if [ ! -f data/curriculum/level2_1house_6goals.json ]; then
    echo "Generating curriculum configs..."
    uv run python modules/envs/scripts/generate_curriculum.py
fi

# Level 2: 1 house (fK2vEV32Lag), 6 goal categories
# 2.4M steps, single scene with diverse goals
uv run python modules/r2dreamer/scripts/run_jax_habitat.py \
    --steps 2400000 \
    --prefill 5000 \
    --checkpoint_every 100000 \
    --output_dir "output/r2dreamer-curriculum-l2/run-${SLURM_JOB_ID}" \
    --seed "${SLURM_JOB_ID}" \
    --log_every 250 \
    --wandb_project 3d-vla-objectnav \
    --wandb_name "r2d-L2-buffix-${SLURM_JOB_ID}" \
    --wandb_tags "curriculum,level2,1house,6goals,buffer-fix,rerun" \
    --curriculum_path data/curriculum/level2_1house_6goals.json \
    --curriculum_mode train \
    --val_data data/val_replay/val_200ep.npz \
    --val_loss_every 10000
