#!/bin/bash
#SBATCH --job-name=3d46-external-offline
#SBATCH --partition=gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=output/3d46-external-offline/slurm-%j.out
#SBATCH --error=output/3d46-external-offline/slurm-%j.err

# 3D-46 external (PyTorch) R2Dreamer offline baseline — one of the 3 WP/CP runs.
#
# Trains the original PyTorch R2Dreamer (external/r2dreamer/) offline on the
# WP/CP vector readout, then logs held-out WM metrics directly comparable to the
# JAX 3D-26 numbers. Runs under the EXTERNAL venv (torchrl/tensordict/gymnasium).
#
# Args (via env vars):
#   ENCODER       wp_cp | aggregator   (default wp_cp; 3D-46 is WP/CP only)
#   SEED          0 | 1 | 2            (required)
#   STEPS         total grad steps     (default 500_000)
#   BUFFER_DIR    offline buffer path  (default data/offline_buffer)
#   WANDB_PROJECT W&B project          (default 3d-vla-objectnav-offline-ablation)
#   EXTRA_ARGS    extra CLI for train_external_offline.py (default empty)

set -euo pipefail

: "${SEED:?SEED must be set (0|1|2)}"
ENCODER="${ENCODER:-wp_cp}"
STEPS="${STEPS:-500000}"
BUFFER_DIR="${BUFFER_DIR:-data/offline_buffer}"
WANDB_PROJECT="${WANDB_PROJECT:-3d-vla-objectnav-offline-ablation}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# RUN_TAG lets smoke runs land in a separate namespace (output + W&B name) so
# they never collide with the prod runs the comparison globs over.
RUN_TAG="${RUN_TAG:-}"
OUTPUT_DIR="output/3d46-external-offline${RUN_TAG:+/$RUN_TAG}/${ENCODER}-seed${SEED}/run-${SLURM_JOB_ID}"
WANDB_NAME="ext-${ENCODER}-seed${SEED}${RUN_TAG:+-$RUN_TAG}"
mkdir -p "${OUTPUT_DIR}"

PY="external/r2dreamer/.venv/bin/python"

echo "Job ${SLURM_JOB_ID} on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "Encoder: ${ENCODER}   Seed: ${SEED}   Steps: ${STEPS}"
echo "Buffer:  ${BUFFER_DIR}"
echo "Output:  ${OUTPUT_DIR}"
echo "WANDB:   ${WANDB_PROJECT}"
echo "Commit:  $(git rev-parse HEAD)   external: $(git -C external/r2dreamer rev-parse HEAD 2>/dev/null || echo N/A)"

# shellcheck disable=SC2086
"${PY}" scripts/r2dreamer/train_external_offline.py \
    --encoder "${ENCODER}" \
    --seed "${SEED}" \
    --steps "${STEPS}" \
    --buffer-dir "${BUFFER_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --device cuda:0 \
    --wandb \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-name "${WANDB_NAME}" \
    --wandb-tags "offline-ablation,3d-24,framework:pytorch-external,seed${SEED}${RUN_TAG:+,$RUN_TAG}" \
    ${EXTRA_ARGS}

echo "Done at $(date)"
