#!/bin/bash
#SBATCH --job-name=agg-skip-heads
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=output/smoke/skip-heads-%j.log
#SBATCH --error=output/smoke/skip-heads-%j.log

set -euo pipefail

WORKTREE="/pfs/data6/home/ul/ul_student/ul_hfj15/worktrees/variant-1-aggregator-mlp"
cd "$WORKTREE"
mkdir -p output/smoke

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
TAG="skip-heads-smoke"
RUN_NAME="${TAG}-${TIMESTAMP}"

echo "=== Aggregator-MLP skip-heads smoke ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Dirty: $(git status --porcelain | wc -l) files"
echo "Run name: ${RUN_NAME}"
echo ""

# ----- Step 1: validate compute_heads=False semantically (GPU unit tests) -----
echo "=== [1/3] Unit tests: compute_heads=False semantics ==="
.venv/bin/pytest -q -x \
    "tests/vggt/test_jax_integration.py::TestJAXFeatureExtractorContract::test_compute_heads_false_returns_only_aggregator" \
    "tests/vggt/test_jax_integration.py::TestJAXFeatureExtractorContract::test_compute_heads_false_aggregator_matches_full"

# ----- Step 2: pipeline profile with new code -----
echo ""
echo "=== [2/3] Pipeline profile (skip-heads enabled) ==="
uv run python scripts/profile_pipeline_aggregator_mlp.py \
    --prefill 200 \
    --warmup 20 \
    --measure 100 \
    --out "output/profiling/pipeline_aggregator_mlp_skip_heads_${SLURM_JOB_ID}.json"

# ----- Step 3: real end-to-end training smoke ~20 min -----
echo ""
echo "=== [3/3] End-to-end training smoke ==="
uv run python scripts/r2dreamer/run_jax_habitat_vggt_aggregator_mlp.py \
    --steps 6000 \
    --prefill 200 \
    --seed 42 \
    --log_every 100 \
    --checkpoint_every 4000 \
    --output_dir "output/smoke/${RUN_NAME}" \
    --wandb_name "${RUN_NAME}" \
    --wandb_tags "${TAG},smoke,skip-heads,pool-on-device"

echo ""
echo "=== Done at $(date) ==="
