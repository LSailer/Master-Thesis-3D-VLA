#!/bin/bash
#SBATCH --job-name=r2d-chain-a
#SBATCH --partition=gpu_h100,gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=output/smoke/r2d-chain-a-%j.out
#SBATCH --error=output/smoke/r2d-chain-a-%j.err

set -euo pipefail

# Chain smoke, part A: validates SLURM job chaining + --resume_from for the
# 1M -> 2M full-run plan. Part A trains 600 prefill + 600 steps and writes its
# final checkpoint; part B (submitted with --dependency=afterok on this job)
# resumes that checkpoint and trains 600 -> 2000. Part B's step budget is
# deliberately > batch_steps (1024) past the resume point, because resume
# skips prefill and must refill the replay buffer on-policy before the first
# train_step can fire — the chain smoke exercises exactly that path.

mkdir -p output/smoke

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "Commit: $(git rev-parse --short HEAD), dirty: $(git status --porcelain | wc -l) files"

if [ ! -f data/curriculum/level1_1house_1goal.json ]; then
    uv run python scripts/environments/generate_curriculum.py
fi

RUN_DIR="output/smoke/r2d-chain-a-${SLURM_JOB_ID}"

export WANDB_MODE=offline
export PYTHONFAULTHANDLER=1

uv run python scripts/r2dreamer/run.py habitat-l1-vggt-house-points-pose \
    --steps 600 \
    --prefill 600 \
    --checkpoint_every 1000000 \
    --output_dir "${RUN_DIR}" \
    --seed 42 \
    --log_every 50 \
    --wandb_project 3d-vla-objectnav \
    --wandb_name "r2d-chain-a-${SLURM_JOB_ID}" \
    --wandb_tags "smoke,chain,resume,gpu_h100" \
    --render_resolution 518

CKPT="${RUN_DIR}/checkpoints/step_000000600.pkl"
if [ ! -f "${CKPT}" ]; then
    echo "[FAIL] expected checkpoint missing: ${CKPT}" >&2
    exit 1
fi
echo "=== Chain A PASS === checkpoint: ${CKPT}"
