#!/bin/bash
#SBATCH --job-name=r2d-L1-vggt-resume-smoke
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=output/r2dreamer-curriculum-l1-vggt/slurm-smoke-%j.out
#SBATCH --error=output/r2dreamer-curriculum-l1-vggt/slurm-smoke-%j.err

mkdir -p output/r2dreamer-curriculum-l1-vggt

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Smoke test for the resume mechanism: 5_000 steps past step 900_000.
# - Throwaway --output_dir → does NOT touch run-3957756/metrics.csv or its
#   checkpoint dir.
# - WANDB_MODE=disabled → does NOT pollute the real W&B run 87u0l6dy with
#   smoke data (the wandb_id="must" resume path is therefore NOT exercised
#   here; that path is small and exercised by review + the unit test that
#   passes wandb_project=None).
# - On success the smoke logdir contains:
#     metrics.csv (header-less because resume_step > 0, with rows at step >= 900k)
#     checkpoints/step_000905000.pkl (the run()-final save)
# - Goal: confirm Trainer.__init__ loads agent state, _train_loop offsets to
#   step 900_000, and CSV append-mode write succeeds end-to-end on real GPU.
RESUME_RUN_DIR="output/r2dreamer-curriculum-l1-vggt/run-3957756"
RESUME_CKPT="${RESUME_RUN_DIR}/checkpoints/step_000900000.pkl"
SMOKE_DIR="output/r2dreamer-curriculum-l1-vggt/smoke-${SLURM_JOB_ID}"

if [ ! -f "${RESUME_CKPT}" ]; then
    echo "ERROR: resume checkpoint missing: ${RESUME_CKPT}" >&2
    exit 1
fi

export WANDB_MODE=disabled

uv run python scripts/r2dreamer/run_jax_habitat_vggt.py \
    --steps 905000 \
    --prefill 5000 \
    --checkpoint_every 100000 \
    --output_dir "${SMOKE_DIR}" \
    --seed "${SLURM_JOB_ID}" \
    --log_every 250 \
    --wandb_project 3d-vla-objectnav \
    --wandb_name "r2d-L1-vggt-resume-smoke-${SLURM_JOB_ID}" \
    --wandb_tags "smoke,resume-test" \
    --resume_from "${RESUME_CKPT}" \
    --render_resolution 518

echo "Smoke complete. Inspect ${SMOKE_DIR}/metrics.csv and ${SMOKE_DIR}/checkpoints/"
