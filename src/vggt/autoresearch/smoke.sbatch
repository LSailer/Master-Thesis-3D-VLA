#!/bin/bash
#SBATCH --job-name=vggt-autoresearch-smoke
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=output/autoresearch/smoke-%j.out
#SBATCH --error=output/autoresearch/smoke-%j.err

# 30-min sanity check for the VGGT JAX autoresearch harness.
# Caches the PyTorch baseline, runs the fast parity gate, executes one
# bench_fast pass, and seeds row 1 of results.tsv for the main loop.

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi

mkdir -p output/autoresearch

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "HEAD: $(git log -1 --oneline)"

# Create (or reset) the autoresearch branch idempotently.
TAG=$(date +%b%d | tr A-Z a-z)
BRANCH="autoresearch/vggt-jax-${TAG}"
echo "Creating/resetting branch $BRANCH from feat/vggt-jax-port"
git checkout feat/vggt-jax-port
git checkout -B "$BRANCH"

# Cache PyTorch baseline (.pt_baseline.json).
echo "Caching PyTorch baseline..."
uv run python -m src.vggt.autoresearch.bench_fast --setup

# Fast parity gate.
echo "Running fast parity gate..."
uv run pytest \
    tests/vggt/test_jax_parity.py::TestLevel1WeightTransfer \
    tests/vggt/test_jax_parity.py::TestLevel2SharedBlockParity \
    -q

# Baseline JAX bench run — capture output for parsing.
BASELINE_LOG="output/autoresearch/baseline-${SLURM_JOB_ID}.log"
echo "Running baseline bench_fast, logging to $BASELINE_LOG"
uv run python -m src.vggt.autoresearch.bench_fast > "$BASELINE_LOG" 2>&1
grep "^speedup:\|^jax_median_ms:\|^pt_baseline_ms:\|^peak_mem_mb:" "$BASELINE_LOG"

# Seed results.tsv with baseline row.
TSV=src/vggt/autoresearch/results.tsv
JAX_MS=$(grep "^jax_median_ms:" "$BASELINE_LOG" | awk '{print $2}')
PT_MS=$(grep "^pt_baseline_ms:" "$BASELINE_LOG" | awk '{print $2}')
SPEEDUP=$(grep "^speedup:" "$BASELINE_LOG" | awk '{print $2}')
PEAK_MB=$(grep "^peak_mem_mb:" "$BASELINE_LOG" | awk '{print $2}')
PEAK_GB=$(awk -v mb="$PEAK_MB" 'BEGIN {printf "%.1f", mb/1024}')
COMMIT=$(git rev-parse --short HEAD)

printf "commit\tjax_ms\tpt_ms\tspeedup\tparity_maxabs\tpeak_mem_gb\tstatus\tdescription\n" > "$TSV"
printf "%s\t%s\t%s\t%s\tn/a\t%s\tbaseline\teager fp32 baseline (Step 8 HEAD)\n" \
    "$COMMIT" "$JAX_MS" "$PT_MS" "$SPEEDUP" "$PEAK_GB" >> "$TSV"
echo "Wrote baseline row to $TSV:"
cat "$TSV"
