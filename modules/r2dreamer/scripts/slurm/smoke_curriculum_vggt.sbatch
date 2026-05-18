#!/bin/bash
#SBATCH --job-name=r2d-vggt-smoke
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=output/smoke/r2d-vggt-smoke-%j.out
#SBATCH --error=output/smoke/r2d-vggt-smoke-%j.err

set -euo pipefail

mkdir -p output/smoke

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Dirty: $(git status --porcelain | wc -l) files"

# Smoke for the L1-L4 VGGT submission set, run on the SAME partition as
# production (gpu_h100) to validate the actual deployment path. The prior
# dev_gpu_h100 smoke aborted in habitat_sim's OpenGL renderer during
# prefill — possibly partition-specific.
#
# L1 is sufficient: L1/L2/L3/L4 differ only in the curriculum JSON path.

if [ ! -f data/curriculum/level1_1house_1goal.json ]; then
    echo "Generating curriculum configs..."
    uv run python modules/envs/scripts/generate_curriculum.py
fi

SMOKE_DIR="output/smoke/r2d-vggt-smoke-${SLURM_JOB_ID}"

# Match production env: wandb offline (so no network calls but the wandb
# threads behave normally) instead of WANDB_MODE=disabled which left a
# wandb asyncio thread idling in the prior abort traceback.
export WANDB_MODE=offline
export PYTHONFAULTHANDLER=1

uv run python modules/r2dreamer/scripts/run_jax_habitat_vggt.py \
    --steps 1500 \
    --prefill 500 \
    --checkpoint_every 1000000 \
    --output_dir "${SMOKE_DIR}" \
    --seed 42 \
    --log_every 50 \
    --wandb_project 3d-vla-objectnav \
    --wandb_name "r2d-vggt-smoke-${SLURM_JOB_ID}" \
    --wandb_tags "smoke,vggt,gpu_h100" \
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
