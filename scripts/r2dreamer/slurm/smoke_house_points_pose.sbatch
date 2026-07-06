#!/bin/bash
#SBATCH --job-name=r2d-hpp-smoke
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:15:00
#SBATCH --output=output/smoke/r2d-hpp-smoke-%j.out
#SBATCH --error=output/smoke/r2d-hpp-smoke-%j.err

set -euo pipefail

# Smoke-test for the LIVE per-scene VGGT house-points buffer.
#
# This validates that the live buffer in src/buffer/house_context_pose_buffer.py
# accumulates house points from scratch (no static sidecar file) and is wired
# through VGGTHousePointsPoseObsAdapter into the encoder `vggt_house_points_pose`
# (run-id habitat-l1-vggt-house-points-pose). Unlike the static-sidecar variant,
# NO --static_house_points_path is passed here — the buffer builds up as the
# agent explores the scene.
#
# This encoder requires the VGGT DPT point head (world_points + confidence) every
# step to feed the buffer, so it runs ~2 s/step (~7x the head-free vggt smoke).
# 600 prefill + 600 train = 1200 env steps -> ~40 min wall clock at that rate,
# well over the 15-min "meaningful results" bar and comfortably inside the 75-min
# budget so the run COMPLETES and prints "Smoke PASS". Training (train_step) begins
# once buffer.size >= batch_size*seq_len (=1024), i.e. partway through the train
# loop; total steps are sized so that phase is reached and exercised.

mkdir -p output/smoke

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Dirty: $(git status --porcelain | wc -l) files"

if [ ! -f data/curriculum/level1_1house_1goal.json ]; then
    echo "Generating curriculum configs..."
    uv run python scripts/environments/generate_curriculum.py
fi

SMOKE_DIR="output/smoke/r2d-hpp-smoke-${SLURM_JOB_ID}"

# Match production env: wandb offline (so no network calls but the wandb
# threads behave normally) instead of WANDB_MODE=disabled which left a
# wandb asyncio thread idling in the prior abort traceback.
export WANDB_MODE=offline
export PYTHONFAULTHANDLER=1

uv run python scripts/r2dreamer/run.py habitat-l1-vggt-house-points-pose \
    --steps 600 \
    --prefill 600 \
    --checkpoint_every 1000000 \
    --output_dir "${SMOKE_DIR}" \
    --seed 42 \
    --log_every 50 \
    --wandb_project 3d-vla-objectnav \
    --wandb_name "r2d-hpp-smoke-${SLURM_JOB_ID}" \
    --wandb_tags "smoke,vggt,house-points-pose,live-buffer,gpu_h100" \
    --render_resolution 518

if [ ! -f "${SMOKE_DIR}/metrics.csv" ]; then
    echo "[FAIL] metrics.csv missing in ${SMOKE_DIR}" >&2
    exit 1
fi
if [ "$(wc -l < "${SMOKE_DIR}/metrics.csv")" -lt 5 ]; then
    echo "[FAIL] metrics.csv has fewer than 5 rows in ${SMOKE_DIR}" >&2
    exit 1
fi

echo "=== Smoke PASS ==="
echo "metrics.csv lines: $(wc -l < "${SMOKE_DIR}/metrics.csv")"
echo "Output: ${SMOKE_DIR}"
