#!/bin/bash
#SBATCH --job-name=crafter-comparison
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=output/slurm/crafter-comparison-%j.out
#SBATCH --error=output/slurm/crafter-comparison-%j.err

# Run all 5 Crafter variants (100K steps each) then execute comparison notebook.
# Submit: sbatch modules/dreamerv3/scripts/run_crafter_comparison.sh

set -euo pipefail

mkdir -p output/slurm output/comparison

echo "=== Crafter 5-Way Comparison ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "================================"

STEPS=100000
PREFILL=5000
SEED=0
LOG_EVERY=250
OUTDIR="output/comparison"

# 1. Our JAX DreamerV3 (existing)
OURS_CSV="${OUTDIR}/ours_metrics.csv"
if [ -f "${OURS_CSV}" ]; then
    echo "[1/5] JAX DreamerV3: already exists, skipping"
else
    echo "[1/5] Running JAX DreamerV3..."
    uv run python modules/dreamerv3/scripts/run_ours_crafter.py \
        --steps ${STEPS} --prefill ${PREFILL} --seed ${SEED} \
        --log_every ${LOG_EVERY} --output "${OURS_CSV}"
fi

# 2. Official DreamerV3
OFFICIAL_CSV="${OUTDIR}/official_metrics.csv"
if [ -f "${OFFICIAL_CSV}" ]; then
    echo "[2/5] Official DreamerV3: already exists, skipping"
else
    echo "[2/5] Running Official DreamerV3..."
    uv run python modules/dreamerv3/scripts/run_official_crafter.py \
        --steps ${STEPS} --seed ${SEED} --output "${OFFICIAL_CSV}"
fi

# 3. PyTorch DreamerV3 (r2dreamer repo, rep_loss=dreamer)
echo "[3/5] Running PyTorch DreamerV3 (decoder)..."
uv run python modules/r2dreamer/scripts/run_pytorch_crafter.py \
    --rep_loss dreamer --steps ${STEPS} --seed ${SEED} \
    --output_dir "${OUTDIR}"

# 4. PyTorch R2-Dreamer (r2dreamer repo, rep_loss=r2dreamer)
echo "[4/5] Running PyTorch R2-Dreamer..."
uv run python modules/r2dreamer/scripts/run_pytorch_crafter.py \
    --rep_loss r2dreamer --steps ${STEPS} --seed ${SEED} \
    --output_dir "${OUTDIR}"

# 5. JAX R2-Dreamer (our implementation)
R2_JAX_CSV="${OUTDIR}/r2dreamer_jax_metrics.csv"
if [ -f "${R2_JAX_CSV}" ]; then
    echo "[5/5] JAX R2-Dreamer: already exists, skipping"
else
    echo "[5/5] Running JAX R2-Dreamer..."
    uv run python modules/r2dreamer/scripts/run_jax_crafter.py \
        --steps ${STEPS} --prefill ${PREFILL} --seed ${SEED} \
        --log_every ${LOG_EVERY} --output "${R2_JAX_CSV}"
fi

# Execute comparison notebook
echo ""
echo "=== Executing comparison notebook ==="
uv run jupyter nbconvert --to notebook --execute \
    modules/dreamerv3/notebooks/official_comparison.ipynb --inplace \
    --ExecutePreprocessor.timeout=21600

echo ""
echo "=== Done at $(date) ==="
echo "Results in: ${OUTDIR}/"
