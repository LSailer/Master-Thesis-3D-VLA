#!/bin/bash
#SBATCH --job-name=3d47-offline-buffer
#SBATCH --partition=gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=output/offline-buffer-3d47/slurm-%j.out
#SBATCH --error=output/offline-buffer-3d47/slurm-%j.err

# 3D-47 offline buffer collection. Identical deterministic rollout to the 3D-25
# buffer (same CNN checkpoint + collect-seed 42) but collected with the 3D-47
# code, so the collector emits all THREE readouts in one pass:
#   z_wp_cp.npz (4116) | z_aggregator.npz (3072) | z_aggregator_both.npz (6144)
# Writes to its own OUT_DIR so the existing data/offline_buffer (used by the
# 3D-26 runs) is never touched.
#
# Args (via env vars / argv):
#   CNN_CHECKPOINT  source policy ckpt  (default: 3D-25 CNN baseline @ 1M; argv[1] also accepted)
#   OUT_DIR         buffer output dir   (default data/offline_buffer_3d47)
#   N_STEPS         transitions         (default 400000 — matches the 3D-25 buffer)

set -euo pipefail

mkdir -p output/offline-buffer-3d47

git_common_dir="$(git rev-parse --git-common-dir)"
main_root="$(realpath "${git_common_dir}/..")"

DEFAULT_CNN_CHECKPOINT="output/r2dreamer-curriculum-l1/run-4367942/checkpoints/step_001000000.pkl"
CNN_CHECKPOINT="${CNN_CHECKPOINT:-${1:-${DEFAULT_CNN_CHECKPOINT}}}"
if [ ! -f "${CNN_CHECKPOINT}" ] && [ -f "${main_root}/${CNN_CHECKPOINT}" ]; then
    CNN_CHECKPOINT="${main_root}/${CNN_CHECKPOINT}"
fi
if [ ! -f "${CNN_CHECKPOINT}" ]; then
    echo "Checkpoint not found in worktree or main checkout: ${CNN_CHECKPOINT}" >&2
    echo "Set CNN_CHECKPOINT, pass it as argv[1], or place it at the default path." >&2
    exit 2
fi

OUT_DIR="${OUT_DIR:-data/offline_buffer_3d47}"
N_STEPS="${N_STEPS:-400000}"

echo "Job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Checkpoint: ${CNN_CHECKPOINT}"
echo "Out dir:    ${OUT_DIR}"
echo "N steps:    ${N_STEPS}"
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
    --n-steps "${N_STEPS}" \
    --collect-seed 42 \
    --out-dir "${OUT_DIR}" \
    --curriculum-path data/curriculum/level1_1house_1goal.json \
    --curriculum-mode train \
    --wandb-project 3d-vla-objectnav \
    --wandb-name "3d47-offline-buffer-${SLURM_JOB_ID}" \
    --wandb-tags "offline-buffer,3d-47,cnn-source,vggt-readouts,aggregator-both" \
    --wandb-init-timeout 600 \
    --log-every 1000 \
    --wandb-log-every 1000 \
    --skeleton-flush-every 10000

echo "Done at $(date)"


##RUN Command:
# ./scripts/r2dreamer/slurm/submit_3d47.sh smoke   # 25 min on gpu_h100_short
# ./scripts/r2dreamer/slurm/submit_3d47.sh prod    # 12h on gpu_h100_il, then trains
