#!/bin/bash
#SBATCH --job-name=pipe-report
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=output/pipeline/report-%j.out
#SBATCH --error=output/pipeline/report-%j.err

# Auto-pipeline report step.
# Runs only if verify passed (afterok dependency from launch.sh).
# Renders an HTML report and opens a PR via claude -p (Sonnet).
#
# Required env (passed via --export from launch.sh):
#   EXPERIMENT_NAME, RECAP_PATH, METRICS_PATH, TRAIN_JOB_ID, VERIFY_JOB_ID, BRANCH

set -uo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    cd "$SLURM_SUBMIT_DIR"
fi

mkdir -p output/pipeline output/reports

: "${EXPERIMENT_NAME:?missing EXPERIMENT_NAME}"
: "${RECAP_PATH:?missing RECAP_PATH}"
: "${METRICS_PATH:?missing METRICS_PATH}"
: "${TRAIN_JOB_ID:?missing TRAIN_JOB_ID}"
: "${VERIFY_JOB_ID:?missing VERIFY_JOB_ID}"
: "${BRANCH:?missing BRANCH}"

PROMPT_FILE="scripts/pipeline/report_prompt.md"
[ -f "$PROMPT_FILE" ] || { echo "ERROR: $PROMPT_FILE not found"; exit 2; }

echo "=== Report $EXPERIMENT_NAME (job $SLURM_JOB_ID) ==="
echo "Recap:   $RECAP_PATH"
echo "Metrics: $METRICS_PATH"
echo "Branch:  $BRANCH"
echo "Train:   $TRAIN_JOB_ID  Verify: $VERIFY_JOB_ID"

PROMPT_TEMPLATE="$(cat "$PROMPT_FILE")"
PROMPT="${PROMPT_TEMPLATE//<EXPERIMENT_NAME>/$EXPERIMENT_NAME}"
PROMPT="${PROMPT//<RECAP_PATH>/$RECAP_PATH}"
PROMPT="${PROMPT//<METRICS_PATH>/$METRICS_PATH}"
PROMPT="${PROMPT//<TRAIN_JOB_ID>/$TRAIN_JOB_ID}"
PROMPT="${PROMPT//<VERIFY_JOB_ID>/$VERIFY_JOB_ID}"
PROMPT="${PROMPT//<BRANCH>/$BRANCH}"

# 30-min hard cap. Report renders HTML + opens PR — should finish in <10 min.
timeout 1500 claude -p \
    --model claude-sonnet-4-6 \
    --dangerously-skip-permissions \
    "$PROMPT"
RC=$?

if [ "$RC" -eq 0 ]; then
    echo "Report DONE — PR should be open."
    exit 0
elif [ "$RC" -eq 124 ]; then
    echo "Report TIMEOUT — partial output may exist in output/reports/."
    exit 1
else
    echo "Report FAILED (rc=$RC). Check output/pipeline/report-${SLURM_JOB_ID}.err"
    exit "$RC"
fi
