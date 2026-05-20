#!/bin/bash
#SBATCH --job-name=r2d-L1-rerun-actfix
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=output/slurm/%j.out
#SBATCH --error=output/slurm/%j.err

mkdir -p output/slurm output/runs/r2dreamer-curriculum-l1-rerun-actfix

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

if [ ! -f data/curriculum/level1_1house_1goal.json ]; then
    echo "Generating curriculum configs..."
    uv run python scripts/environments/generate_curriculum.py
fi

# L1-rerun actent-fix rerun — supersedes l1-rerun-buffix (y5a0upzd, 75% SR).
# Cleanest actent-isolation experiment: buffer fix was already in the
# predecessor, this run isolates the actent=3e-4 vs 3e-2 effect.
uv run python scripts/r2dreamer/run_jax_habitat.py \
    --steps 2400000 \
    --prefill 5000 \
    --checkpoint_every 100000 \
    --output_dir "output/runs/r2dreamer-curriculum-l1-rerun-actfix/run-${SLURM_JOB_ID}" \
    --seed "${SLURM_JOB_ID}" \
    --log_every 250 \
    --act_entropy 3e-4 \
    --wandb_project 3d-vla-objectnav \
    --wandb_name "r2d-L1-rerun-actfix-${SLURM_JOB_ID}" \
    --wandb_tags "curriculum,level1,1house,chair-only,buffer-fix,actent-fix,actfix-rerun" \
    --val_data data/val_replay/val_200ep.npz \
    --val_loss_every 10000
