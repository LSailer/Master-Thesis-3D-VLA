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
[ -f modules/vggt/autoresearch/.pt_baseline.json ] || { echo "ERROR: .pt_baseline.json missing - run smoke.sbatch first"; exit 1; }
[ -f modules/vggt/autoresearch/results.tsv ] || { echo "ERROR: results.tsv missing - run smoke.sbatch first"; exit 1; }
[ "$(wc -l < modules/vggt/autoresearch/results.tsv)" -ge 2 ] || { echo "ERROR: results.tsv has no baseline row"; exit 1; }
CUR_BRANCH=$(git rev-parse --abbrev-ref HEAD)
case "$CUR_BRANCH" in
    autoresearch/vggt-jax-*) echo "On branch $CUR_BRANCH, OK";;
    *) echo "ERROR: must run from an autoresearch/vggt-jax-* branch, currently on $CUR_BRANCH"; exit 1;;
esac

# Outer loop: call claude -p until time runs out.
TIME_LIMIT_SECONDS=100000   # 27.7h, leaves 2.3h headroom on 30h allocation
ITER=0
PROMPT="Read modules/vggt/autoresearch/program.md and execute ONE iteration of the experiment loop. Commit or reset based on results. Return when the iteration is complete."

while [ $SECONDS -lt $TIME_LIMIT_SECONDS ]; do
    ITER=$((ITER+1))
    echo ""
    echo "=== Iteration $ITER at $(date) (elapsed: ${SECONDS}s) ==="
    if claude -p --dangerously-skip-permissions "$PROMPT"; then
        echo "Iteration $ITER returned 0"
    else
        RC=$?
        echo "Iteration $ITER returned non-zero ($RC) - continuing"
    fi
done

echo ""
echo "=== Loop ended after $ITER iterations, $SECONDS seconds ==="

# Summary.
TSV=modules/vggt/autoresearch/results.tsv
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
