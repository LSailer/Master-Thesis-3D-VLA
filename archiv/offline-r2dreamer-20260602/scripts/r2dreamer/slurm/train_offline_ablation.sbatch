#!/bin/bash
#SBATCH --job-name=3d26-offline-ablation
#SBATCH --partition=gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=output/3d26-offline-ablation/slurm-%j.out
#SBATCH --error=output/3d26-offline-ablation/slurm-%j.err

# 3D-26 Offline R2Dreamer ablation — one of the 6 runs (variant × seed).
#
# Args (via env vars):
#   ENCODER       wp_cp | aggregator   (required)
#   SEED          0 | 1 | 2            (required)
#   STEPS         total grad steps     (default 500_000)
#   BUFFER_DIR    offline buffer path  (default data/offline_buffer)
#   WANDB_PROJECT W&B project          (default 3d-vla-objectnav-offline-ablation)
#   EXTRA_ARGS    extra CLI for train_offline_ablation.py (default empty)

set -euo pipefail

: "${ENCODER:?ENCODER must be set (wp_cp|aggregator)}"
: "${SEED:?SEED must be set (0|1|2)}"
STEPS="${STEPS:-500000}"
BUFFER_DIR="${BUFFER_DIR:-data/offline_buffer}"
WANDB_PROJECT="${WANDB_PROJECT:-3d-vla-objectnav-offline-ablation}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

OUTPUT_DIR="output/3d26-offline-ablation/${ENCODER}-seed${SEED}/run-${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p output/3d26-offline-ablation

git_common_dir="$(git rev-parse --git-common-dir)"
main_root="$(realpath "${git_common_dir}/..")"

echo "Job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Encoder: ${ENCODER}"
echo "Seed:    ${SEED}"
echo "Steps:   ${STEPS}"
echo "Buffer:  ${BUFFER_DIR}"
echo "Output:  ${OUTPUT_DIR}"
echo "WANDB project: ${WANDB_PROJECT}"
echo "Commit:  $(git rev-parse HEAD)"
echo "Dirty:   $(git status --porcelain | wc -l) files"

./scripts/setup_worktree.sh

# Mirror the VGGT external-repo links used by the collection job — even though
# the offline trainer never calls the VGGT extractor, the encoders package
# imports JAXVGGTFeatureExtractor at module load.
mkdir -p external
for name in InfiniteVGGT StreamVGGT VGGT; do
    target="${main_root}/external/${name}"
    link="external/${name}"
    if [ ! -e "${link}" ] && [ -e "${target}" ]; then
        ln -s "$(realpath --relative-to external "${target}")" "${link}"
    fi
done

WANDB_NAME="3d26-${ENCODER}-seed${SEED}-${SLURM_JOB_ID}"

# shellcheck disable=SC2086
.venv/bin/python scripts/r2dreamer/train_offline_ablation.py \
    --encoder "${ENCODER}" \
    --seed "${SEED}" \
    --steps "${STEPS}" \
    --buffer-dir "${BUFFER_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-name "${WANDB_NAME}" \
    --wandb-tags "3d-26,offline-ablation,${ENCODER},seed${SEED}" \
    ${EXTRA_ARGS}

echo "Done at $(date)"
