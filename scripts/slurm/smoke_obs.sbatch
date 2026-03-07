#!/bin/bash
#SBATCH --job-name=smoke-obs
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=output/slurm/%j.out
#SBATCH --error=output/slurm/%j.err

# Minimal wandb observation collection — 5 episodes, 100 steps.
# Usage: sbatch --partition=dev_gpu_h100 scripts/slurm/smoke_obs.sbatch

set -euo pipefail
mkdir -p output/slurm

# Activate uv venv
source .venv/bin/activate

# Wandb config
export WANDB_PROJECT=3d-vla-objectnav
export WANDB_RUN_GROUP="${SLURM_JOB_NAME}"
export WANDB_TAGS="cluster,smoke,${SLURM_JOB_PARTITION:-unknown}"

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
python -c "import habitat_sim; print(f'habitat-sim {habitat_sim.__version__}')"

# Run 5 episodes x 100 steps — minimal check that obs + wandb work
python scripts/collect_random_obs.py \
    --num-episodes 5 \
    --max-steps 100 \
    --log-image-every 20 \
    --no-save

echo "=== Done ==="
