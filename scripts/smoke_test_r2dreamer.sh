#!/usr/bin/env bash
# Smoke test for R2-Dreamer Habitat training pipeline.
# Exercises the exact same entry points as the real SLURM job.
#
# Usage:
#   srun --partition=dev_gpu_h100 --gres=gpu:1 --time=00:15:00 \
#       bash scripts/smoke_test_r2dreamer.sh
#
# What it tests:
#   1. Val data collection (collect_val_data.py)
#   2. R2-Dreamer training with val loss (run_jax_habitat.py)
#   3. Checkpoint evaluation with semantic + topdown (eval_habitat.py)
#   4. Output file assertions

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTDIR="output/smoke-r2dreamer-${TIMESTAMP}"
VAL_DATA="${OUTDIR}/val_data.npz"

echo "============================================"
echo "  R2-Dreamer Smoke Test"
echo "  Output: ${OUTDIR}"
echo "============================================"

# ------------------------------------------------------------------
# Step 1: Collect val data (5 episodes, random actions)
# ------------------------------------------------------------------
echo ""
echo "=== Step 1/4: Collect val replay data ==="
uv run python scripts/environments/collect_val_data.py \
    --episodes 5 \
    --split val \
    --max_geodesic 5.0 \
    --max_episode_steps 200 \
    --obs_size 64 \
    --output "$VAL_DATA"

if [ ! -f "$VAL_DATA" ]; then
    echo "[FAIL] Val data not created at $VAL_DATA"
    exit 1
fi
echo "[PASS] Val data collected"

# ------------------------------------------------------------------
# Step 2: Train R2-Dreamer (500 steps, val loss at 250)
# ------------------------------------------------------------------
echo ""
echo "=== Step 2/4: R2-Dreamer training with val loss ==="
TRAIN_DIR="${OUTDIR}/train"
uv run python scripts/r2dreamer/run_jax_habitat.py \
    --steps 500 \
    --prefill 200 \
    --output_dir "$TRAIN_DIR" \
    --seed 42 \
    --log_every 100 \
    --checkpoint_every 250 \
    --wandb_project dreamerv3-objectnav \
    --wandb_name "smoke-r2dreamer-${TIMESTAMP}" \
    --val_data "$VAL_DATA" \
    --val_loss_every 250

# Check training outputs
if [ ! -f "${TRAIN_DIR}/metrics.csv" ]; then
    echo "[FAIL] metrics.csv not created"
    exit 1
fi
if ! ls "${TRAIN_DIR}/checkpoints/"*.pkl 1>/dev/null 2>&1; then
    echo "[FAIL] No checkpoint files created"
    exit 1
fi
# Check val loss was logged
if ! grep -q "val/total_loss" "${TRAIN_DIR}/metrics.csv"; then
    echo "[FAIL] val/total_loss not found in metrics.csv"
    exit 1
fi
echo "[PASS] Training completed with val loss"

# ------------------------------------------------------------------
# Step 3: Eval with semantic + topdown
# ------------------------------------------------------------------
echo ""
echo "=== Step 3/4: Checkpoint evaluation (semantic + topdown) ==="
CHECKPOINT=$(ls -t "${TRAIN_DIR}/checkpoints/"*.pkl | head -1)
EVAL_OUTPUT="${OUTDIR}/eval_results.json"
uv run python scripts/r2dreamer/eval_habitat.py \
    --checkpoint "$CHECKPOINT" \
    --episodes 2 \
    --split val \
    --semantic \
    --render_topdown \
    --output "$EVAL_OUTPUT"

if [ ! -f "$EVAL_OUTPUT" ]; then
    echo "[FAIL] Eval results not created at $EVAL_OUTPUT"
    exit 1
fi
if ! ls "${OUTDIR}/topdown/"*.png 1>/dev/null 2>&1; then
    echo "[FAIL] No topdown PNG files created"
    exit 1
fi
echo "[PASS] Evaluation completed with semantic + topdown"

# ------------------------------------------------------------------
# Step 4: Validate outputs
# ------------------------------------------------------------------
echo ""
echo "=== Step 4/4: Validating outputs ==="

# Count val loss entries in metrics
VAL_COUNT=$(grep -c "val/total_loss" "${TRAIN_DIR}/metrics.csv" || true)
echo "  Val loss entries: ${VAL_COUNT}"
if [ "$VAL_COUNT" -lt 1 ]; then
    echo "[FAIL] Expected at least 1 val loss entry"
    exit 1
fi

# Check eval results have episodes
EP_COUNT=$(python3 -c "
import json
with open('${EVAL_OUTPUT}') as f:
    data = json.load(f)
print(len(data['results']))
")
echo "  Eval episodes: ${EP_COUNT}"
if [ "$EP_COUNT" -lt 2 ]; then
    echo "[FAIL] Expected 2 eval episodes, got ${EP_COUNT}"
    exit 1
fi

# Count topdown maps
TD_COUNT=$(ls "${OUTDIR}/topdown/"*.png 2>/dev/null | wc -l)
echo "  Topdown maps: ${TD_COUNT}"

echo "[PASS] All output validation passed"

echo ""
echo "============================================"
echo "  R2-DREAMER SMOKE TEST PASSED"
echo "  Output: ${OUTDIR}"
echo "============================================"
