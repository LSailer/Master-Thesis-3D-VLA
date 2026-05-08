#!/bin/bash
#SBATCH --job-name=r2d-ab-external-smoke
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=output/runs/r2dreamer-parity-crafter/external-smoke-%j.out
#SBATCH --error=output/runs/r2dreamer-parity-crafter/external-smoke-%j.err

mkdir -p output/runs/r2dreamer-parity-crafter

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Smoke test: 2k env steps to validate the run_external_crafter.py wrapper
# (wandb.init(sync_tensorboard=True) + Hydra invocation + external venv).
# Goal: catch wrapper bugs before committing 24h to the full external A/B run.

EXT_DIR="$PWD/external/r2dreamer"
VENV="$EXT_DIR/.venv"

if [ ! -d "$VENV" ]; then
    echo "ERROR: external venv not found at $VENV"
    exit 1
fi

source "$VENV/bin/activate"

LOGDIR="$PWD/output/runs/r2dreamer-parity-crafter/external-smoke-${SLURM_JOB_ID}"
mkdir -p "$LOGDIR"

# wandb.mode=offline so the smoke doesn't pollute the W&B project.
python modules/r2dreamer/launch/run_external_crafter.py \
    --wandb_project "r2dreamer-parity-crafter" \
    --wandb_name "external-smoke-${SLURM_JOB_ID}" \
    --wandb_tags "smoke,external,crafter" \
    --wandb_mode offline \
    --logdir "$LOGDIR" \
    --seed 0 \
    --env crafter \
    env.steps=2000 \
    env.eval_episode_num=0 \
    model=size12M \
    model.compile=False \
    device=cuda:0 \
    buffer.storage_device=cuda:0
