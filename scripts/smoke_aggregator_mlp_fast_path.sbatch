#!/bin/bash
#SBATCH --job-name=agg-mlp-smoke
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=output/smoke/agg-mlp-fast-path-%j.log
#SBATCH --error=output/smoke/agg-mlp-fast-path-%j.log

set -euo pipefail

WORKTREE="/pfs/data6/home/ul/ul_student/ul_hfj15/worktrees/variant-1-aggregator-mlp"
cd "$WORKTREE"

mkdir -p output/smoke

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
TAG="agg-mlp-fast-path-smoke"
RUN_NAME="${TAG}-${TIMESTAMP}"

echo "=== Aggregator-MLP fast-path smoke ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Dirty: $(git status --porcelain | wc -l) files"
echo "Run name: ${RUN_NAME}"
echo ""

uv run python modules/r2dreamer/scripts/run_jax_habitat_vggt_aggregator_mlp.py \
    --steps 800 \
    --prefill 200 \
    --seed 42 \
    --log_every 50 \
    --checkpoint_every 10000 \
    --output_dir "output/smoke/${RUN_NAME}" \
    --wandb_name "${RUN_NAME}" \
    --wandb_tags "${TAG},smoke,short-15min,pool-on-device"

echo ""
echo "=== Done at $(date) ==="
