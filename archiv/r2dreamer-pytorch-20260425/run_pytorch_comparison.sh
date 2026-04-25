#!/bin/bash
#SBATCH --job-name=pytorch-r2dreamer
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=output/slurm/pytorch-r2dreamer-%j.out
#SBATCH --error=output/slurm/pytorch-r2dreamer-%j.err

# Run PyTorch r2dreamer on Crafter (both dreamer and r2dreamer modes, 100K steps each)
# then re-execute the comparison notebook.
# Submit: sbatch modules/r2dreamer/scripts/run_pytorch_comparison.sh

set -euo pipefail
mkdir -p output/slurm output/comparison

echo "=== PyTorch R2-Dreamer Crafter Comparison ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "============================================="

STEPS=100000
SEED=0
OUTDIR="output/comparison"

# 1. PyTorch DreamerV3 (rep_loss=dreamer, with decoder)
echo "[1/2] Running PyTorch DreamerV3 (decoder)..."
uv run python modules/r2dreamer/scripts/run_pytorch_crafter.py \
    --rep_loss dreamer --steps ${STEPS} --seed ${SEED} \
    --output_dir "${OUTDIR}"

# 2. PyTorch R2-Dreamer (rep_loss=r2dreamer, no decoder)
echo "[2/2] Running PyTorch R2-Dreamer..."
uv run python modules/r2dreamer/scripts/run_pytorch_crafter.py \
    --rep_loss r2dreamer --steps ${STEPS} --seed ${SEED} \
    --output_dir "${OUTDIR}"

# Re-execute the comparison notebook with all data
echo ""
echo "=== Executing comparison notebook ==="
uv run jupyter nbconvert --to notebook --execute \
    modules/dreamerv3/notebooks/official_comparison.ipynb --inplace \
    --ExecutePreprocessor.timeout=21600

echo ""
echo "=== Done at $(date) ==="
ls -la ${OUTDIR}/*metrics.csv ${OUTDIR}/*.png 2>/dev/null
