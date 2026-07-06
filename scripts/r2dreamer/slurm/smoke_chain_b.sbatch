#!/bin/bash
#SBATCH --job-name=r2d-chain-b
#SBATCH --partition=gpu_h100,gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --output=output/smoke/r2d-chain-b-%j.out
#SBATCH --error=output/smoke/r2d-chain-b-%j.err

set -euo pipefail

# Chain smoke, part B: resumes part A's final checkpoint and continues
# training to step 2000. Submit with:
#   sbatch --dependency=afterok:<A_JOBID> \
#          --export=ALL,RESUME_DIR=output/smoke/r2d-chain-a-<A_JOBID> \
#          scripts/r2dreamer/slurm/smoke_chain_b.sbatch
#
# PASS criteria (checked below):
#   1. trainer prints "Resumed agent state ... at step 600"
#   2. trainer prints "Resume mode: skipping prefill"
#   3. run completes with final checkpoint step_000002000.pkl
# Together these prove the exact mechanism the 1M -> 2M production chain uses.

: "${RESUME_DIR:?RESUME_DIR must be exported (part A output dir)}"

# Latest checkpoint from part A (robust to periodic checkpoints if enabled).
CKPT="$(ls -1 "${RESUME_DIR}"/checkpoints/step_*.pkl | sort | tail -1)"
echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "Resuming from: ${CKPT}"

RUN_DIR="output/smoke/r2d-chain-b-${SLURM_JOB_ID}"

export WANDB_MODE=offline
export PYTHONFAULTHANDLER=1

uv run python scripts/r2dreamer/run.py habitat-l1-vggt-house-points-pose \
    --steps 2000 \
    --prefill 600 \
    --resume_from "${CKPT}" \
    --checkpoint_every 1000000 \
    --output_dir "${RUN_DIR}" \
    --seed 43 \
    --log_every 50 \
    --wandb_project 3d-vla-objectnav \
    --wandb_name "r2d-chain-b-${SLURM_JOB_ID}" \
    --wandb_tags "smoke,chain,resume,gpu_h100" \
    --render_resolution 518 \
    2>&1 | tee "output/smoke/r2d-chain-b-${SLURM_JOB_ID}.train.log"

LOG="output/smoke/r2d-chain-b-${SLURM_JOB_ID}.train.log"
FAIL=0
grep -q "Resumed agent state" "${LOG}" || { echo "[FAIL] no 'Resumed agent state' line" >&2; FAIL=1; }
grep -q "Resume mode: skipping prefill" "${LOG}" || { echo "[FAIL] no 'skipping prefill' line" >&2; FAIL=1; }
[ -f "${RUN_DIR}/checkpoints/step_000002000.pkl" ] || { echo "[FAIL] final checkpoint step_000002000.pkl missing" >&2; FAIL=1; }
[ "${FAIL}" -eq 0 ] || exit 1
echo "=== Chain B PASS === resume + continue verified"
