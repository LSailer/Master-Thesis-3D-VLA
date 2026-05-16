#!/bin/bash
# Protocol E: same overfit-one-batch test for cnn and vggt encoders.
#
# Pairs with diag_overfit_one_batch.sbatch (which runs the aggregator-MLP
# variant). Identical hyper-params, seed, batch shape and loss weights so
# the three runs are directly comparable.
#
# If cnn and vggt both drive loss to ~0 but vggt_aggregator_mlp plateaus,
# the regression is isolated to the mean-pool + linear-projection design,
# not VGGT itself and not the training infra.
#
# Submit: sbatch scripts/diag_overfit_cnn_vggt.sbatch
#
#SBATCH --job-name=enc-diag-cnn-vggt
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=output/diag/enc-diag-cnn-vggt-%j.log
#SBATCH --error=output/diag/enc-diag-cnn-vggt-%j.log

set -euo pipefail

WORKTREE="/pfs/data6/home/ul/ul_student/ul_hfj15/worktrees/variant-1-aggregator-mlp"
cd "$WORKTREE"
mkdir -p output/diag

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
TAG="enc-diag"

echo "=== Encoder overfit-one-batch comparison (CNN + VGGT) ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Dirty: $(git status --porcelain | wc -l) files"
echo ""

# Identical to diag_overfit_one_batch.sbatch — fair comparison.
COMMON=(
    --prefill 200
    --steps 1
    --seed 42
    --log_every 25
    --checkpoint_every 1000000
    --overfit_one_batch
    --overfit_steps 500
    --overfit_batch_size 1
    --overfit_seq_len 8
    --actor_loss_weight 0
    --value_loss_weight 0
    --repval_loss_weight 0
    --lr 3e-4
    --batch_size 1
    --seq_len 8
)

# ------------------------------------------------------------------ Run 1
RUN1="${TAG}-cnn-${TIMESTAMP}"
echo "--- Run 1/2: CNN encoder ---"
uv run python modules/r2dreamer/scripts/run_jax_habitat.py \
    "${COMMON[@]}" \
    --output_dir "output/diag/${RUN1}" \
    --wandb_name "${RUN1}" \
    --wandb_tags "diag,overfit,wm-only,encoder-cnn,protocol-e"

# ------------------------------------------------------------------ Run 2
RUN2="${TAG}-vggt-${TIMESTAMP}"
echo ""
echo "--- Run 2/2: VGGT (wp_cp) encoder ---"
uv run python modules/r2dreamer/scripts/run_jax_habitat_vggt.py \
    "${COMMON[@]}" \
    --output_dir "output/diag/${RUN2}" \
    --wandb_name "${RUN2}" \
    --wandb_tags "diag,overfit,wm-only,encoder-vggt,protocol-e"

echo ""
echo "=== Done at $(date) ==="
echo "Compare loss/* curves and params/encoder_l2 across:"
echo "  CNN:     ${RUN1}"
echo "  VGGT:    ${RUN2}"
echo "  AggMLP:  agg-mlp-diag-overfit-stopgrad-* (from job 4719993)"
