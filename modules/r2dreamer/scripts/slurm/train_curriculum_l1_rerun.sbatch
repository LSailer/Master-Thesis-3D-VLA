#!/bin/bash
#SBATCH --job-name=r2d-L1-rerun-buffix
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=output/r2dreamer-curriculum-l1-rerun/slurm-%j.out
#SBATCH --error=output/r2dreamer-curriculum-l1-rerun/slurm-%j.err

mkdir -p output/r2dreamer-curriculum-l1-rerun

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Generate curriculum configs if not present
if [ ! -f data/curriculum/level1_1house_1goal.json ]; then
    echo "Generating curriculum configs..."
    uv run python modules/envs/scripts/generate_curriculum.py
fi

# Level 1 re-run with buffer bug fixes:
# - write-head crossing fix (no corrupted sequences after buffer wraps)
# - ema_state checkpoint fix (return normalization preserved on resume)
# Same config as original L1 baseline (2.4M steps, 1 house, chair only)
uv run python modules/r2dreamer/scripts/run_jax_habitat.py \
    --steps 2400000 \
    --prefill 5000 \
    --checkpoint_every 100000 \
    --output_dir "output/r2dreamer-curriculum-l1-rerun/run-${SLURM_JOB_ID}" \
    --seed "${SLURM_JOB_ID}" \
    --log_every 250 \
    --wandb_project 3d-vla-objectnav \
    --wandb_name "r2d-L1-buffix-${SLURM_JOB_ID}" \
    --wandb_tags "curriculum,level1,1house,chair-only,buffer-fix,rerun" \
    --val_data data/val_replay/val_200ep.npz \
    --val_loss_every 10000
