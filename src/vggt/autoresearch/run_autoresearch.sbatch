#!/bin/bash
#SBATCH --job-name=vggt-autoresearch-main
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --output=output/autoresearch/run-%j.out
#SBATCH --error=output/autoresearch/run-%j.err

# Main 30h VGGT JAX autoresearch loop.
# Prerequisites: smoke.sbatch must have been run (.pt_baseline.json cached,
# row 1 of results.tsv seeded, on an autoresearch/vggt-jax-* branch).

set -uo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi

mkdir -p output/autoresearch

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
echo "HEAD: $(git log -1 --oneline)"

# Fail fast if smoke.sbatch hasn't run.
[ -f src/vggt/autoresearch/.pt_baseline.json ] || { echo "ERROR: .pt_baseline.json missing - run smoke.sbatch first"; exit 1; }
[ -f src/vggt/autoresearch/results.tsv ] || { echo "ERROR: results.tsv missing - run smoke.sbatch first"; exit 1; }
[ "$(wc -l < src/vggt/autoresearch/results.tsv)" -ge 2 ] || { echo "ERROR: results.tsv has no baseline row"; exit 1; }
CUR_BRANCH=$(git rev-parse --abbrev-ref HEAD)
case "$CUR_BRANCH" in
    autoresearch/vggt-jax-*) echo "On branch $CUR_BRANCH, OK";;
    *) echo "ERROR: must run from an autoresearch/vggt-jax-* branch, currently on $CUR_BRANCH"; exit 1;;
esac

# Outer loop: call claude -p until time runs out.
# TIME_LIMIT_SECONDS can be overridden via --export=ALL,TIME_LIMIT_SECONDS=... at submit time
# (e.g. 1500 for a 25-min shake-out on dev_gpu_h100). Default 100000 = 27.7h on a 30h allocation.
TIME_LIMIT_SECONDS="${TIME_LIMIT_SECONDS:-100000}"
ITER=0
PROMPT="Read src/vggt/autoresearch/program.md and execute ONE iteration of the experiment loop. Commit or reset based on results. Return when the iteration is complete."

ITER_TIMEOUT_SECONDS="${ITER_TIMEOUT_SECONDS:-900}"   # 15 min hard cap per iteration
while [ $SECONDS -lt $TIME_LIMIT_SECONDS ]; do
    ITER=$((ITER+1))
    ITER_LOG="output/autoresearch/iter-${SLURM_JOB_ID}-${ITER}.log"
    echo ""
    echo "=== Iteration $ITER at $(date) (elapsed: ${SECONDS}s, log: $ITER_LOG) ==="
    if timeout "$ITER_TIMEOUT_SECONDS" claude -p --dangerously-skip-permissions "$PROMPT" > "$ITER_LOG" 2>&1; then
        echo "Iteration $ITER returned 0"
    else
        RC=$?
        if [ "$RC" -eq 124 ]; then
            echo "Iteration $ITER TIMEOUT after ${ITER_TIMEOUT_SECONDS}s - continuing (see $ITER_LOG)"
        else
            echo "Iteration $ITER returned non-zero ($RC) - continuing (see $ITER_LOG)"
        fi
    fi
done

echo ""
echo "=== Loop ended after $ITER iterations, $SECONDS seconds ==="

# Summary.
TSV=src/vggt/autoresearch/results.tsv
echo "Results summary:"
echo "  total rows:    $(($(wc -l < "$TSV") - 1))"
echo "  keep:          $(grep -c '	keep	' "$TSV" || true)"
echo "  discard:       $(grep -c '	discard	' "$TSV" || true)"
echo "  crash_parity:  $(grep -c '	crash_parity	' "$TSV" || true)"
echo "  crash_bench:   $(grep -c '	crash_bench	' "$TSV" || true)"
echo "  crash_full:    $(grep -c '	crash_full	' "$TSV" || true)"
echo ""
echo "Best speedup so far:"
awk -F'\t' '$7=="keep" {print $4, $1, $8}' "$TSV" | sort -rn | head -5
