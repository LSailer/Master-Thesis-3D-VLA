#!/bin/bash
# scripts/pipeline/launch.sh <name>
#
# Submit the auto-pipeline SLURM chain for experiment <name>.
# Reads scripts/pipeline/<name>.args (written by /engineer-team), submits:
#   train.sbatch                            -> JOB_T
#   verify.sbatch  --dependency=afterok:T   -> JOB_V  (claude -p verifier)
#   report.sbatch  --dependency=afterok:V   -> JOB_R  (claude -p reporter, opens PR)
#
# verify exits non-zero on failure -> report does NOT run; verify itself runs gh issue create.

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <experiment-name>" >&2
    exit 2
fi

NAME="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ARGS_FILE="$SCRIPT_DIR/${NAME}.args"

[ -f "$ARGS_FILE" ] || { echo "ERROR: missing $ARGS_FILE" >&2; exit 1; }

# Source the args (sets EXPERIMENT_NAME, RECAP_PATH, TRAIN_PARTITION, TRAIN_TIME, TRAIN_CMD, METRICS_PATH)
# shellcheck disable=SC1090
source "$ARGS_FILE"

: "${EXPERIMENT_NAME:?args file must set EXPERIMENT_NAME}"
: "${RECAP_PATH:?args file must set RECAP_PATH}"
: "${TRAIN_PARTITION:?args file must set TRAIN_PARTITION}"
: "${TRAIN_TIME:?args file must set TRAIN_TIME}"
: "${TRAIN_CMD:?args file must set TRAIN_CMD}"
: "${METRICS_PATH:?args file must set METRICS_PATH}"

cd "$REPO_ROOT"

# Sanity: recap exists, branch is pipeline/*, repo clean except for pipeline scaffolding
[ -f "$RECAP_PATH" ] || { echo "ERROR: recap $RECAP_PATH not found" >&2; exit 1; }
CUR_BRANCH=$(git rev-parse --abbrev-ref HEAD)
case "$CUR_BRANCH" in
    pipeline/*) ;;
    *) echo "ERROR: must run from a pipeline/* branch (currently: $CUR_BRANCH)" >&2; exit 1;;
esac

mkdir -p output/pipeline output/slurm

# --- 1) Train job
JOB_T=$(sbatch \
    --parsable \
    --partition="$TRAIN_PARTITION" \
    --time="$TRAIN_TIME" \
    --job-name="pipe-train-$NAME" \
    --export=ALL,EXPERIMENT_NAME="$NAME",TRAIN_CMD="$TRAIN_CMD" \
    scripts/slurm/train.sbatch)

echo "Submitted train job: $JOB_T"

# --- 2) Verify job — depends on train success
JOB_V=$(sbatch \
    --parsable \
    --dependency="afterok:$JOB_T" \
    --kill-on-invalid-dep=yes \
    --job-name="pipe-verify-$NAME" \
    --export=ALL,EXPERIMENT_NAME="$NAME",RECAP_PATH="$RECAP_PATH",METRICS_PATH="$METRICS_PATH",TRAIN_JOB_ID="$JOB_T" \
    "$SCRIPT_DIR/verify.sbatch")

echo "Submitted verify job: $JOB_V (after train $JOB_T)"

# --- 3) Report job — depends on verify success (so failure = no report, only gh issue from verify)
JOB_R=$(sbatch \
    --parsable \
    --dependency="afterok:$JOB_V" \
    --kill-on-invalid-dep=yes \
    --job-name="pipe-report-$NAME" \
    --export=ALL,EXPERIMENT_NAME="$NAME",RECAP_PATH="$RECAP_PATH",METRICS_PATH="$METRICS_PATH",TRAIN_JOB_ID="$JOB_T",VERIFY_JOB_ID="$JOB_V",BRANCH="$CUR_BRANCH" \
    "$SCRIPT_DIR/report.sbatch")

echo "Submitted report job: $JOB_R (after verify $JOB_V)"

cat <<EOF

Pipeline launched for: $NAME
  Branch:  $CUR_BRANCH
  Train:   $JOB_T  ($TRAIN_PARTITION, $TRAIN_TIME)
  Verify:  $JOB_V  (claude -p sonnet, runs after train)
  Report:  $JOB_R  (claude -p sonnet, runs after verify pass — opens PR)

On failure, verify creates a GitHub issue and report is skipped (afterok dependency).
Monitor with:  squeue -u \$USER
EOF
