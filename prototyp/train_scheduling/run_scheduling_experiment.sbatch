#!/bin/bash
#SBATCH --job-name=r2d-train-sched
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=uc3n089
#SBATCH --output=output/train-sched-%j.out
#SBATCH --error=output/train-sched-%j.err

# Smoke comparison of the three train-scheduling modes (prototyp/
# train_scheduling/IDEA.md): interleaved (production credit scheme), episode
# (train at episode boundaries), threaded (actor + learner threads over a
# LockedReplayBuffer). One process per mode, shared results dir, judged by
# outputs/prototype/train_scheduling/run-$SLURM_JOB_ID/MANIFEST.json — NOT by
# exit codes (habitat GL teardown SIGABRTs on this cluster; the runner
# hard-exits after writing its manifest entry).
#
# Deliberately .venv/bin/python (NOT `uv run`): prod-mode uv re-syncs the
# shared venv and can break habitat under a bumped numpy (see MEMORY).

set -uo pipefail

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Commit: $(git rev-parse --short HEAD)"

scripts/slurm/hooks/link_external.sh

if [ ! -f data/curriculum/level1_1house_1goal.json ]; then
    echo "Generating curriculum configs..."
    .venv/bin/python scripts/environments/generate_curriculum.py
fi

OUT="outputs/prototype/train_scheduling/run-${SLURM_JOB_ID}"
mkdir -p "$OUT"

# Smoke shape mirrors the validated YAML smoke configs (B=4 T=16 ratio=16);
# 600 steps after a 200-step prefill keeps each mode a few minutes on H100
# while crossing at least one L1 episode boundary for the episode mode.
SMOKE_FLAGS="--steps 600 --prefill 200 --batch_size 4 --seq_len 16 \
  --train_ratio 16 --log_every 100 --seed 42"

for MODE in interleaved episode threaded; do
    echo "=== mode: $MODE ==="
    .venv/bin/python prototyp/train_scheduling/run_scheduling_experiment.py \
        --mode "$MODE" --run_id habitat-l1-cnn --out_dir "$OUT" \
        $SMOKE_FLAGS || echo "mode $MODE exited non-zero (check manifest)"
done

echo "=== summary ==="
.venv/bin/python prototyp/train_scheduling/run_scheduling_experiment.py \
    --summarize "$OUT" || true

echo "=== done; judge by $OUT/MANIFEST.json ==="
cat "$OUT/MANIFEST.json" || true
