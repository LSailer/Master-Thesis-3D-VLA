#!/bin/bash
#SBATCH --job-name=3d-35-verify-metrics
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:28:00
#SBATCH --output=output/verify-3d-35/slurm-%j.log
#SBATCH --error=output/verify-3d-35/slurm-%j.log

# Verification run for the 3D-35 metrics overhaul. Dev-sized CNN + L1
# training on dev_gpu_h100 (30-min cap). In-run validation and episode-video
# rendering were removed from training; use src.r2dreamer.launch.evaluate on
# saved checkpoints for held-out eval metrics/videos.
#
# Submit from the project root:
#   sbatch scripts/verify_3d_35_metrics.sbatch
#
# When the run starts, watch the W&B dashboard the SLURM stdout prints.

set -euo pipefail

# Run from wherever sbatch was submitted so worktrees + main checkout both work.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p output/verify-3d-35

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
TAG="verify-3d-35"
RUN_NAME="${TAG}-cnn-l1-${TIMESTAMP}"

echo "=== 3D-35 metrics-overhaul verification (15k steps, CNN, L1, no in-run val/video, dev_gpu_h100) ==="
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date:    $(date)"
echo "Branch:  $(git branch --show-current)"
echo "Commit:  $(git rev-parse --short HEAD)"
echo "Dirty:   $(git status --porcelain | wc -l) files"
echo "Run:     ${RUN_NAME}"
echo ""

uv run python -m src.main train \
    --env habitat \
    --encoder cnn \
    --curriculum L1 \
    --steps 15000 \
    --prefill 2000 \
    --seed 0 \
    --log_every 250 \
    --checkpoint_every 5000 \
    --act_entropy 3e-2 \
    --output_dir "output/verify-3d-35/${RUN_NAME}" \
    --wandb_name "${RUN_NAME}" \
    --wandb_tags "3D-35,metrics-overhaul,verify,cnn,L1"

echo ""
echo "=== Done at $(date) ==="
echo ""
echo "Verify in W&B:"
echo "  - train/ metrics are logged during training"
echo "  - NO val/metrics/* and NO episode-video keys are emitted by train"
echo "  - run offline evaluate from a checkpoint for held-out eval metrics/videos"
echo "  - NO kl_prior / kl_post duplicates (3D-42 — already absent in current code)"
