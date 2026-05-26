#!/bin/bash
#SBATCH --job-name=3d27-online-eval
#SBATCH --partition=gpu_h100_il
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=output/3d27-online-eval/slurm-%j.out
#SBATCH --error=output/3d27-online-eval/slurm-%j.err

# 3D-27 Online navigation evaluation for one 3D-26 offline-ablation checkpoint.
#
# Args (via env vars):
#   ENCODER          wp_cp | aggregator                 (required)
#   SEED             0 | 1 | 2                          (required)
#   CHECKPOINT       explicit checkpoint path           (optional)
#   CHECKPOINT_ROOT  root containing 3D-26 run outputs   (default output/3d26-offline-ablation)
#   EPISODES         HM3D val episodes to evaluate      (default 50)
#   EVAL_SEED        evaluation RNG seed                (default SEED)
#   SPLIT            Habitat split                      (default val)
#   WANDB_PROJECT    W&B project for eval videos        (default empty/off)
#   EXTRA_ARGS       extra CLI for src.main evaluate    (default empty)

set -euo pipefail

: "${ENCODER:?ENCODER must be set (wp_cp|aggregator)}"
: "${SEED:?SEED must be set (0|1|2)}"
case "${ENCODER}" in wp_cp|aggregator) ;; *) echo "bad ENCODER=${ENCODER}" >&2; exit 2 ;; esac
case "${SEED}" in 0|1|2) ;; *) echo "bad SEED=${SEED}" >&2; exit 2 ;; esac

export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-output/3d26-offline-ablation}"
EPISODES="${EPISODES:-50}"
EVAL_SEED="${EVAL_SEED:-${SEED}}"
SPLIT="${SPLIT:-val}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

case "${ENCODER}" in
    wp_cp) EVAL_ENCODER="vggt" ;;
    aggregator) EVAL_ENCODER="vggt_aggregator_mlp" ;;
esac

mkdir -p output/3d27-online-eval
./scripts/setup_worktree.sh

git_common_dir="$(git rev-parse --git-common-dir)"
main_root="$(realpath "${git_common_dir}/..")"

# Mirror the VGGT external-repo links needed by the online feature extractor.
mkdir -p external
for name in InfiniteVGGT StreamVGGT VGGT; do
    target="${main_root}/external/${name}"
    link="external/${name}"
    if [ ! -e "${link}" ] && [ -e "${target}" ]; then
        ln -s "$(realpath --relative-to external "${target}")" "${link}"
    fi
done

if [ -z "${CHECKPOINT:-}" ]; then
    CHECKPOINT="$(python - <<'PY'
import os
from pathlib import Path
root = Path(os.environ["CHECKPOINT_ROOT"])
encoder = os.environ["ENCODER"]
seed = os.environ["SEED"]
base = root / f"{encoder}-seed{seed}"
ckpts = sorted(base.glob("run-*/checkpoints/step_*.pkl"))
if not ckpts:
    raise SystemExit(f"no checkpoints found under {base}/run-*/checkpoints/step_*.pkl")
# Prefer the numerically largest training step; tie-break by mtime for reruns.
def key(path: Path):
    step = int(path.stem.split("_")[-1])
    return (step, path.stat().st_mtime)
print(max(ckpts, key=key))
PY
)"
fi

OUTPUT_DIR="output/3d27-online-eval/${ENCODER}-seed${SEED}/run-${SLURM_JOB_ID}"
mkdir -p "${OUTPUT_DIR}"

WANDB_ARGS=()
if [ -n "${WANDB_PROJECT}" ]; then
    WANDB_ARGS=(--wandb_project "${WANDB_PROJECT}" --wandb_name "3d27-${ENCODER}-seed${SEED}-${SLURM_JOB_ID}")
fi

cat <<EOF
Job ${SLURM_JOB_ID} on $(hostname) at $(date)
GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)
Encoder:          ${ENCODER} (${EVAL_ENCODER})
Train seed:       ${SEED}
Eval seed:        ${EVAL_SEED}
Episodes:         ${EPISODES}
Split:            ${SPLIT}
Checkpoint:       ${CHECKPOINT}
Output:           ${OUTPUT_DIR}
Commit:           $(git rev-parse HEAD)
Dirty files:      $(git status --porcelain | wc -l)
EOF

# shellcheck disable=SC2086
.venv/bin/python -m src.main evaluate \
    --env habitat \
    --encoder "${EVAL_ENCODER}" \
    --checkpoint "${CHECKPOINT}" \
    --episodes "${EPISODES}" \
    --seed "${EVAL_SEED}" \
    --split "${SPLIT}" \
    --output_dir "${OUTPUT_DIR}" \
    "${WANDB_ARGS[@]}" \
    ${EXTRA_ARGS}

echo "Done at $(date)"
