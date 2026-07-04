#!/bin/bash
#SBATCH --job-name=r2d-cmp-mlp
#SBATCH --partition=gpu_h100,gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=output/compare/r2d-cmp-mlp-%j.out
#SBATCH --error=output/compare/r2d-cmp-mlp-%j.err

set -euo pipefail

# SR/SPL comparison arm B: production MLP house branch
# (HousePointsCameraEncoder). Must stay flag-identical to
# compare_gnn_house_50k.sbatch except the run-id and names.

mkdir -p output/compare

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Dirty: $(git status --porcelain | wc -l) files"

if [ ! -f data/curriculum/level1_1house_1goal.json ]; then
    echo "Generating curriculum configs..."
    uv run python scripts/environments/generate_curriculum.py
fi

RUN_DIR="output/compare/r2d-cmp-mlp-${SLURM_JOB_ID}"

export WANDB_MODE=offline
export PYTHONFAULTHANDLER=1

uv run python scripts/r2dreamer/run.py habitat-l1-vggt-house-points-pose \
    --steps 50000 \
    --prefill 5000 \
    --checkpoint_every 25000 \
    --output_dir "${RUN_DIR}" \
    --seed 42 \
    --log_every 250 \
    --val_every 25000 \
    --val_episodes 10 \
    --wandb_project 3d-vla-objectnav \
    --wandb_name "r2d-cmp-mlp-${SLURM_JOB_ID}" \
    --wandb_tags "compare,sr-spl,vggt,house-points-pose,mlp,live-buffer,gpu_h100" \
    --render_resolution 518

echo "=== run finished at $(date) ==="
echo "Output: ${RUN_DIR}"
