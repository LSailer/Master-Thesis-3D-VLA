#!/bin/bash
#SBATCH --job-name=3dvla-invariance
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=output/slurm/%j.out
#SBATCH --error=output/slurm/%j.err

# Usage: sbatch --partition=gpu_h100 --time=01:00:00 scripts/r2dreamer/invariance_extract.sbatch
# Extra args after the script are forwarded to invariance_extract.py, e.g.:
#   sbatch ... scripts/r2dreamer/invariance_extract.sbatch --num-episodes 5 --max-steps 30

set -euo pipefail
mkdir -p output/slurm

# The active 3.10 venv has the right nvidia dist-info pins (cudnn 9.10.2.21,
# cusparselt 0.7.1, nccl, nvshmem) but their .so payloads were stripped out.
# We restored them at the matching versions into a sidecar dir (one-time:
#   uv pip install --target ~/.cuda-sidecar-310 \
#     nvidia-cudnn-cu12==9.10.2.21 nvidia-cusparselt-cu12==0.7.1 \
#     nvidia-nccl-cu12==2.27.5 nvidia-nvshmem-cu12==3.4.5
# ). Expose ONLY those four (NOT the sidecar's cublas/nvrtc, which are 12.9 and
# would clash with the venv's 12.8 build that torch/jax link against).
SIDE="$HOME/.cuda-sidecar-310/nvidia"
export LD_LIBRARY_PATH="$SIDE/cudnn/lib:$SIDE/cusparselt/lib:$SIDE/nccl/lib:$SIDE/nvshmem/lib:${LD_LIBRARY_PATH:-}"

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
uv run python -c "import jax; print(f'JAX devices: {jax.devices()}')"

uv run python scripts/r2dreamer/invariance_extract.py "$@"
