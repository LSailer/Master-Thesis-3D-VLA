#!/bin/bash
#SBATCH --job-name=r2d-gnnhpp-smoke
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:15:00
#SBATCH --output=output/smoke/r2d-gnnhpp-smoke-%j.out
#SBATCH --error=output/smoke/r2d-gnnhpp-smoke-%j.err

set -euo pipefail

# Smoke-test for the prototype GNN house branch (src/prototyp/gnn_house_encoder).
#
# Identical to smoke_house_points_pose.sbatch except the run-id: same live
# per-scene house buffer, same VGGT point head, but the agent-side house branch
# is a k-NN graph + GCN over a 4096-node even-stride subset of the snapshot
# instead of the per-point MLP + global pool. Pass criteria match the MLP smoke
# so step time and losses are directly comparable.

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

SMOKE_DIR="output/smoke/r2d-gnnhpp-smoke-${SLURM_JOB_ID}"

export WANDB_MODE=offline
export PYTHONFAULTHANDLER=1

uv run python scripts/r2dreamer/run.py habitat-l1-gnn-house-points-pose \
    --steps 600 \
    --prefill 600 \
    --checkpoint_every 1000000 \
    --output_dir "${SMOKE_DIR}" \
    --seed 42 \
    --log_every 50 \
    --wandb_project 3d-vla-objectnav \
    --wandb_name "r2d-gnnhpp-smoke-${SLURM_JOB_ID}" \
    --wandb_tags "smoke,vggt,house-points-pose,gnn,live-buffer,gpu_h100" \
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
