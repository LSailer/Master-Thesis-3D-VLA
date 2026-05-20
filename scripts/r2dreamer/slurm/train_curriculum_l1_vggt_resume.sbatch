#!/bin/bash
#SBATCH --job-name=r2d-L1-vggt-resume
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --output=output/r2dreamer-curriculum-l1-vggt/slurm-%j.out
#SBATCH --error=output/r2dreamer-curriculum-l1-vggt/slurm-%j.err

mkdir -p output/r2dreamer-curriculum-l1-vggt

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Resume L1 VGGT run 87u0l6dy from step 900_000 to 2_400_000.
# - Reuses run-3957756 logdir → checkpoints land alongside step_000{100..900}000.pkl
#   and metrics.csv is appended (not clobbered).
# - --wandb_id 87u0l6dy + resume="must" reattaches to the same W&B run; the
#   chart continues without a split. Fails loudly if the id is wrong.
# - Skips the random prefill: trained policy refills the buffer until it
#   contains >= batch_size*seq_len transitions, then training resumes.
RESUME_RUN_DIR="output/r2dreamer-curriculum-l1-vggt/run-3957756"
RESUME_CKPT="${RESUME_RUN_DIR}/checkpoints/step_000900000.pkl"

if [ ! -f "${RESUME_CKPT}" ]; then
    echo "ERROR: resume checkpoint missing: ${RESUME_CKPT}" >&2
    exit 1
fi

uv run python scripts/r2dreamer/run_jax_habitat_vggt.py \
    --steps 2400000 \
    --prefill 5000 \
    --checkpoint_every 100000 \
    --output_dir "${RESUME_RUN_DIR}" \
    --seed "${SLURM_JOB_ID}" \
    --log_every 250 \
    --wandb_project 3d-vla-objectnav \
    --wandb_name "r2d-L1-vggt-resume-${SLURM_JOB_ID}" \
    --wandb_tags "curriculum,level1,1house,chair-only,vggt,3d-encoder,resume-from-900k" \
    --wandb_id 87u0l6dy \
    --resume_from "${RESUME_CKPT}" \
    --render_resolution 518
