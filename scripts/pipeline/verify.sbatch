#!/bin/bash
#SBATCH --job-name=pipe-verify
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=output/pipeline/verify-%j.out
#SBATCH --error=output/pipeline/verify-%j.err

# Auto-pipeline verify step.
# Reads recap criteria + metrics, judges pass/fail via claude -p (Sonnet).
# Pass: writes results into recap frontmatter, exits 0 -> report runs.
# Fail: creates GitHub issue, exits 1 -> report is skipped (afterok).
#
# Required env (passed via --export from launch.sh):
#   EXPERIMENT_NAME, RECAP_PATH, METRICS_PATH, TRAIN_JOB_ID

set -uo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi

mkdir -p output/pipeline

: "${EXPERIMENT_NAME:?missing EXPERIMENT_NAME}"
: "${RECAP_PATH:?missing RECAP_PATH}"
: "${METRICS_PATH:?missing METRICS_PATH}"
: "${TRAIN_JOB_ID:?missing TRAIN_JOB_ID}"

PROMPT_FILE="scripts/pipeline/verify_prompt.md"
[ -f "$PROMPT_FILE" ] || { echo "ERROR: $PROMPT_FILE not found"; exit 2; }

echo "=== Verify $EXPERIMENT_NAME (job $SLURM_JOB_ID, train was $TRAIN_JOB_ID) ==="
echo "Recap:   $RECAP_PATH"
echo "Metrics: $METRICS_PATH"

PROMPT_TEMPLATE="$(cat "$PROMPT_FILE")"
PROMPT="${PROMPT_TEMPLATE//<EXPERIMENT_NAME>/$EXPERIMENT_NAME}"
PROMPT="${PROMPT//<RECAP_PATH>/$RECAP_PATH}"
PROMPT="${PROMPT//<METRICS_PATH>/$METRICS_PATH}"
PROMPT="${PROMPT//<TRAIN_JOB_ID>/$TRAIN_JOB_ID}"

# 30-min hard cap. Verify is structured judgment + maybe one gh write — should finish in <10 min.
timeout 1500 claude -p \
    --model claude-sonnet-4-6 \
    --dangerously-skip-permissions \
    "$PROMPT"
RC=$?

if [ "$RC" -eq 0 ]; then
    echo "Verify PASSED — report will run."
    exit 0
elif [ "$RC" -eq 124 ]; then
    echo "Verify TIMEOUT — treating as failure, report will be skipped."
    exit 1
else
    echo "Verify FAILED (rc=$RC) — gh issue should have been created. Report will be skipped."
    exit 1
fi
