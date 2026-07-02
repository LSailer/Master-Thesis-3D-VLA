#!/bin/bash
#SBATCH --job-name=r2d-hpp-1m
#SBATCH --partition=gpu_h100,gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-23:59:00
#SBATCH --output=output/full/r2d-hpp-1m-%j.out
#SBATCH --error=output/full/r2d-hpp-1m-%j.err

set -euo pipefail

# Full-scale run of the LIVE house-points buffer encoder: 1M environment steps
# (5k prefill + 1M train) with the static 262,144-point zero-padded snapshot
# and masked pooling (HOUSE_CONTEXT_SIZE_KEY). Goals:
#   1. Real training at thesis scale on the new full-buffer encoder path.
#   2. Read the end-of-run "house buffer summary" (total_points,
#      max_fill_fraction, overflow, points-growth curve) to see how much of
#      the 2^23 buffer a complete run actually uses.
#
# Sized from smoke job 5706115 (same encoder, H100): ~158 ms/step steady state
# -> 1M steps ~ 44 h. 2M steps (~88 h) does NOT fit any partition MaxTime in
# one job, and the end-of-run summary only prints on completion, so we run 1M
# here; extend to 2M afterwards with --resume_from on the last checkpoint
# (written every 50k steps).
#
# Dual partition gpu_h100,gpu_h100_il: whichever frees first takes the job.
# gpu_h100_il caps MaxTime at 2-00:00:00, hence --time just under 48 h
# (~4 h margin over the 44 h estimate). If the wall is ever hit, the 50k-step
# checkpoints + the afterany-chained part-2 job make it self-healing.
#
# Smoke also showed the map reaches 3.38M points / 40% buffer fill after only
# 1200 env steps -- far above HOUSE_CONTEXT_MAX_POINTS, so the snapshot's
# even-stride fallback is engaged for most of this run. The growth curve from
# this job decides whether to bump the constant or investigate cross-episode
# frame misalignment before the 2M extension.

mkdir -p output/full

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Dirty: $(git status --porcelain | wc -l) files"

if [ ! -f data/curriculum/level1_1house_1goal.json ]; then
    echo "Generating curriculum configs..."
    uv run python scripts/environments/generate_curriculum.py
fi

RUN_DIR="output/full/r2d-hpp-1m-${SLURM_JOB_ID}"

export WANDB_MODE=offline
export PYTHONFAULTHANDLER=1

uv run python scripts/r2dreamer/run.py habitat-l1-vggt-house-points-pose \
    --steps 1000000 \
    --prefill 5000 \
    --checkpoint_every 50000 \
    --output_dir "${RUN_DIR}" \
    --seed 42 \
    --log_every 250 \
    --wandb_project 3d-vla-objectnav \
    --wandb_name "r2d-hpp-1m-${SLURM_JOB_ID}" \
    --wandb_tags "full,vggt,house-points-pose,live-buffer,262k-snapshot,gpu_h100" \
    --render_resolution 518

echo "=== run finished at $(date) ==="
echo "Output: ${RUN_DIR}"
