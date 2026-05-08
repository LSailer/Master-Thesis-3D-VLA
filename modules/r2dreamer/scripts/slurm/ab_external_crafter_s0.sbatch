#!/bin/bash
#SBATCH --job-name=r2d-ab-external-crafter
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=output/runs/r2dreamer-parity-crafter/external-%j.out
#SBATCH --error=output/runs/r2dreamer-parity-crafter/external-%j.err

mkdir -p output/runs/r2dreamer-parity-crafter

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# A/B with modules/r2dreamer (JAX): both write to project r2dreamer-parity-crafter.
# External impl: NM512's PyTorch reference (ICLR 2026), R2-Dreamer rep_loss, size12M model.
# Step budget 1.01e6 (upstream default for env=crafter).
# wandb logging via the run_external_crafter.py wrapper (sync_tensorboard=True),
# which avoids editing the vendored external/ tree.

EXT_DIR="$PWD/external/r2dreamer"
VENV="$EXT_DIR/.venv"

if [ ! -d "$VENV" ]; then
    echo "ERROR: external venv not found at $VENV"
    echo "Run: bash modules/r2dreamer/scripts/setup_external_venv.sh"
    exit 1
fi

# Activate the dedicated external venv (separate from parent uv project to
# avoid numpy/torch pin conflicts).
source "$VENV/bin/activate"

LOGDIR="$PWD/output/runs/r2dreamer-parity-crafter/external-${SLURM_JOB_ID}"
mkdir -p "$LOGDIR"

python modules/r2dreamer/launch/run_external_crafter.py \
    --wandb_project "r2dreamer-parity-crafter" \
    --wandb_name "external-torch-crafter-ref-s0-${SLURM_JOB_ID}" \
    --wandb_tags "ab,parity,external,torch,crafter,r2dreamer" \
    --logdir "$LOGDIR" \
    --seed 0 \
    --env crafter \
    model=size12M \
    model.rep_loss=r2dreamer \
    model.compile=True \
    device=cuda:0 \
    buffer.storage_device=cuda:0
