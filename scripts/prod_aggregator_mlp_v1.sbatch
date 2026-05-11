#!/bin/bash
#SBATCH --job-name=agg-mlp-prod-v1
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=42:00:00
#SBATCH --output=output/prod/agg-mlp-prod-v1-%j.log
#SBATCH --error=output/prod/agg-mlp-prod-v1-%j.log

set -euo pipefail

WORKTREE="/pfs/data6/home/ul/ul_student/ul_hfj15/worktrees/variant-1-aggregator-mlp"
cd "$WORKTREE"
mkdir -p output/prod

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
TAG="agg-mlp-prod-v1"
RUN_NAME="${TAG}-${TIMESTAMP}"

echo "=== Aggregator-MLP production run v1 (42h, 2M steps) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Dirty: $(git status --porcelain | wc -l) files"
echo "Run name: ${RUN_NAME}"
echo ""

uv run python modules/r2dreamer/scripts/run_jax_habitat_vggt_aggregator_mlp.py \
    --steps 2000000 \
    --prefill 5000 \
    --seed 42 \
    --log_every 1000 \
    --checkpoint_every 50000 \
    --output_dir "output/prod/${RUN_NAME}" \
    --wandb_name "${RUN_NAME}" \
    --wandb_tags "${TAG},pool-on-device,skip-heads,prod-42h"

echo ""
echo "=== Done at $(date) ==="
