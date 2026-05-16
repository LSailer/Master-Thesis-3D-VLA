#!/bin/bash
#SBATCH --job-name=agg-pipeline-profile
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=output/profiling/pipeline-aggregator-mlp-%j.log
#SBATCH --error=output/profiling/pipeline-aggregator-mlp-%j.log

set -euo pipefail

WORKTREE="/pfs/data6/home/ul/ul_student/ul_hfj15/worktrees/variant-1-aggregator-mlp"
cd "$WORKTREE"
mkdir -p output/profiling

echo "=== Aggregator-MLP pipeline profile ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Dirty: $(git status --porcelain | wc -l) files"
echo ""

uv run python scripts/profile_pipeline_aggregator_mlp.py \
    --prefill 200 \
    --warmup 20 \
    --measure 100 \
    --out "output/profiling/pipeline_aggregator_mlp_${SLURM_JOB_ID}.json"

echo ""
echo "=== Done at $(date) ==="
