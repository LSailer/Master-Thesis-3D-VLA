#!/bin/bash
#SBATCH --job-name=3d25-offline-buffer
#SBATCH --partition=gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=output/offline-buffer-3d25/slurm-%j.out
#SBATCH --error=output/offline-buffer-3d25/slurm-%j.err

set -euo pipefail

mkdir -p output/offline-buffer-3d25

git_common_dir="$(git rev-parse --git-common-dir)"
main_root="$(realpath "${git_common_dir}/..")"

DEFAULT_CNN_CHECKPOINT="output/r2dreamer-curriculum-l1/run-4367942/checkpoints/step_001000000.pkl"
CNN_CHECKPOINT="${CNN_CHECKPOINT:-${1:-}}"
if [ -z "${CNN_CHECKPOINT}" ] && [ -f "${DEFAULT_CNN_CHECKPOINT}" ]; then
    CNN_CHECKPOINT="${DEFAULT_CNN_CHECKPOINT}"
fi
if [ -n "${CNN_CHECKPOINT}" ] && [ ! -f "${CNN_CHECKPOINT}" ] && [ -f "${main_root}/${CNN_CHECKPOINT}" ]; then
    CNN_CHECKPOINT="${main_root}/${CNN_CHECKPOINT}"
fi
if [ -z "${CNN_CHECKPOINT}" ] || [ ! -f "${CNN_CHECKPOINT}" ]; then
    echo "Set CNN_CHECKPOINT or pass the CNN baseline @ 1M checkpoint as argv[1]." >&2
    echo "Example:" >&2
    echo "  sbatch $0 output/r2dreamer-curriculum-l1/run-4367942/checkpoints/step_001000000.pkl" >&2
    exit 2
fi

echo "Job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Checkpoint: ${CNN_CHECKPOINT}"
echo "Commit: $(git rev-parse HEAD)"
echo "Dirty files: $(git status --porcelain | wc -l)"

./scripts/setup_worktree.sh

# The JAX VGGT loader resolves StreamVGGT relative to this worktree. These
# heavy external repos are gitignored in the main checkout, so make local links
# when the job is launched from a fresh worktree.
mkdir -p external
for name in InfiniteVGGT StreamVGGT VGGT; do
    target="${main_root}/external/${name}"
    link="external/${name}"
    if [ ! -e "${link}" ] && [ -e "${target}" ]; then
        ln -s "$(realpath --relative-to external "${target}")" "${link}"
    fi
done

.venv/bin/python scripts/r2dreamer/collect_offline_buffer.py \
    --checkpoint "${CNN_CHECKPOINT}" \
    --n-steps 500000 \
    --collect-seed 42 \
    --out-dir data/offline_buffer \
    --wandb-project 3d-vla-objectnav \
    --wandb-name "3d25-offline-buffer-${SLURM_JOB_ID}" \
    --wandb-tags "offline-buffer,3d-25,cnn-source,vggt-readouts" \
    --wandb-init-timeout 600 \
    --log-every 1000 \
    --wandb-log-every 1000

echo "Done at $(date)"
