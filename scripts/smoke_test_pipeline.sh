#!/usr/bin/env bash
# Smoke test pipeline: exercises every code path in train.py before long runs.
#
# Usage:
#   srun --partition=dev_gpu_h100 --gres=gpu:1 --time=00:30:00 bash scripts/smoke_test_pipeline.sh
#
# What it tests:
#   1. Imports + GPU (scripts/smoke_test.py)
#   2. Training with periodic eval mid-training (eval->train transition)
#   3. Checkpoint save
#   4. Checkpoint resume + continued training
#   5. wandb metric assertions (training, eval, resume)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
RUN_NAME="smoke-pipeline-${TIMESTAMP}"
RESUME_NAME="smoke-resume-${TIMESTAMP}"
LOGDIR="output/smoke-pipeline-${TIMESTAMP}"
WANDB_PROJECT="dreamerv3-objectnav"

echo "============================================"
echo "  Smoke Test Pipeline"
echo "  Run name: ${RUN_NAME}"
echo "  Logdir:   ${LOGDIR}"
echo "============================================"

# ------------------------------------------------------------------
# Step 1: Pre-check (imports + GPU)
# ------------------------------------------------------------------
echo ""
echo "=== Step 1/4: Pre-check (imports + GPU) ==="
uv run python scripts/smoke_test.py
echo "[PASS] Pre-check passed"

# ------------------------------------------------------------------
# Step 2: Training with periodic eval
#   - 1000 total steps, eval at step 500, save at step 500
#   - eval_episodes=2 (fast), prefill=200
#   - This exercises: prefill, training, logging, eval, eval->train
#     transition, checkpoint save
# ------------------------------------------------------------------
echo ""
echo "=== Step 2/4: Training with periodic eval ==="
uv run python -m src.dreamerv3.train \
    --total_steps 1000 \
    --prefill_steps 200 \
    --eval_every 500 \
    --eval_episodes 2 \
    --save_every 500 \
    --log_every 100 \
    --batch_size 4 \
    --seq_len 16 \
    --buffer_capacity 10000 \
    --split val_mini \
    --max_episode_steps 200 \
    --obs_size 64 \
    --max_geodesic 5.0 \
    --logdir "$LOGDIR" \
    --wandb_name "$RUN_NAME" \
    --wandb_tags "smoke-test,pipeline"

echo "[PASS] Training completed"

# ------------------------------------------------------------------
# Step 3: Checkpoint resume
#   - Load from the checkpoint saved at step 500
#   - Run another 500 steps with eval
# ------------------------------------------------------------------
echo ""
echo "=== Step 3/4: Checkpoint resume ==="
uv run python -m src.dreamerv3.train \
    --total_steps 500 \
    --prefill_steps 100 \
    --eval_every 250 \
    --eval_episodes 2 \
    --save_every 500 \
    --log_every 100 \
    --batch_size 4 \
    --seq_len 16 \
    --buffer_capacity 10000 \
    --split val_mini \
    --max_episode_steps 200 \
    --obs_size 64 \
    --max_geodesic 5.0 \
    --logdir "${LOGDIR}-resume" \
    --checkpoint "$LOGDIR" \
    --wandb_name "$RESUME_NAME" \
    --wandb_tags "smoke-test,pipeline,resume"

echo "[PASS] Checkpoint resume completed"

# ------------------------------------------------------------------
# Step 4: Validate wandb metrics
# ------------------------------------------------------------------
echo ""
echo "=== Step 4/4: Validating wandb metrics ==="
uv run python scripts/validate_smoke_wandb.py \
    --project "$WANDB_PROJECT" \
    --run_name "$RUN_NAME" \
    --resume_name "$RESUME_NAME"

echo ""
echo "============================================"
echo "  SMOKE TEST PIPELINE PASSED"
echo "============================================"
