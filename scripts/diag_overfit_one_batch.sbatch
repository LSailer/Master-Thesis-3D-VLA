#!/bin/bash
# Karpathy step-3 diagnostic for the aggregator-MLP encoder.
#
# Runs two short overfit-one-batch jobs back-to-back on dev_gpu_h100:
#   (1) Protocol B+C: WM-only (actor/value/repval=0), default stop_grad
#   (2) Protocol B+C+D: same, plus barlow_grad_to_encoder
#
# Each run prefills a small replay (B*T=8 transitions need very few steps),
# samples one batch, freezes it, and runs --overfit_steps train_step
# iterations. If loss/{dyn,rep,rew,con,barlow} does not drop monotonically
# on the frozen batch, the gradient path is broken and no 42 h production
# run will save it.
#
# Submit: sbatch scripts/diag_overfit_one_batch.sbatch
#
#SBATCH --job-name=agg-mlp-diag-overfit
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=output/diag/agg-mlp-diag-overfit-%j.log
#SBATCH --error=output/diag/agg-mlp-diag-overfit-%j.log

set -euo pipefail

WORKTREE="/pfs/data6/home/ul/ul_student/ul_hfj15/worktrees/variant-1-aggregator-mlp"
cd "$WORKTREE"
mkdir -p output/diag

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
TAG="agg-mlp-diag-overfit"

echo "=== Aggregator-MLP overfit-one-batch diagnostic ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Dirty: $(git status --porcelain | wc -l) files"
echo ""

COMMON=(
    --prefill 200              # buffer-size lower bound is overfit_batch_size*overfit_seq_len = 8
    --steps 1                  # ignored in overfit mode; keep small to avoid total_steps work
    --seed 42
    --log_every 25
    --checkpoint_every 1000000 # disable checkpointing
    --overfit_one_batch
    --overfit_steps 500
    --overfit_batch_size 1
    --overfit_seq_len 8
    # Protocol C: world model only.
    --actor_loss_weight 0
    --value_loss_weight 0
    --repval_loss_weight 0
    # Hint: a larger LR is fine for an overfit memorise run.
    --lr 3e-4
    --batch_size 1
    --seq_len 8
)

# ------------------------------------------------------------------ Run 1
RUN1="${TAG}-stopgrad-${TIMESTAMP}"
echo "--- Run 1/2: default stop_gradient (current behaviour) ---"
uv run python scripts/r2dreamer/run_jax_habitat_vggt_aggregator_mlp.py \
    "${COMMON[@]}" \
    --output_dir "output/diag/${RUN1}" \
    --wandb_name "${RUN1}" \
    --wandb_tags "diag,overfit,wm-only,stopgrad,protocol-bc"

# ------------------------------------------------------------------ Run 2
RUN2="${TAG}-bargrad-${TIMESTAMP}"
echo ""
echo "--- Run 2/2: Barlow gradient reaches encoder (Protocol D) ---"
uv run python scripts/r2dreamer/run_jax_habitat_vggt_aggregator_mlp.py \
    "${COMMON[@]}" \
    --barlow_grad_to_encoder \
    --output_dir "output/diag/${RUN2}" \
    --wandb_name "${RUN2}" \
    --wandb_tags "diag,overfit,wm-only,bargrad,protocol-bcd"

echo ""
echo "=== Done at $(date) ==="
echo "Compare loss/* curves and params/encoder_l2 between:"
echo "  ${RUN1}"
echo "  ${RUN2}"
