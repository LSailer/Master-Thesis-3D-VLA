#!/bin/bash
#SBATCH --job-name=r2d-L3d3-p5split
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=output/runs/duell3-l3-p5-split/slurm-%j.out
#SBATCH --error=output/runs/duell3-l3-p5-split/slurm-%j.err

mkdir -p output/runs/duell3-l3-p5-split

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
GPU_MEMORY_LOG="output/runs/duell3-l3-p5-split/gpu-memory-${SLURM_JOB_ID}.csv"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu --format=csv -l 5 > "$GPU_MEMORY_LOG" 2>/dev/null &
    GPU_MONITOR_PID=$!
    trap 'if [ -n "${GPU_MONITOR_PID:-}" ]; then kill "$GPU_MONITOR_PID" 2>/dev/null || true; wait "$GPU_MONITOR_PID" 2>/dev/null || true; fi' EXIT
    echo "GPU memory log: $GPU_MEMORY_LOG"
fi

export R2DREAMER_HARD_EXIT_ON_FINISH="1"
export SEED="42"

scripts/slurm/hooks/link_external.sh

# Generate curriculum configs if not present
if [ ! -f data/curriculum/level3_10houses_1goal.json ]; then
    echo "Generating curriculum configs..."
    uv run python scripts/environments/generate_curriculum.py
fi

# Duell 3 (2026-07-29-r3) P5: the P1 blocks as three separate 2048 fields (camera, patch mean, patch max), one MLP branch each plus the composite fusion Dense, instead of one 6144 -> 1024 projection (adapter aggregator_pooled_full_split). Same information and replay cost as P1; measures whether per-block encoder capacity beats one wide projection. Training knobs frozen to C.
uv run --no-sync python -m src.main \
    --env habitat \
    --steps 1500000 \
    --adapter aggregator_pooled_full_split \
    --curriculum L3 \
    --prefill 1024 \
    --checkpoint_every 100000 \
    --output_dir "output/runs/duell3-l3-p5-split/run-${SLURM_JOB_ID}" \
    --seed "${SEED}" \
    --log_every 250 \
    --buffer_capacity 500000 \
    --wandb_project 3d-vla-objectnav \
    --wandb_name "duell3-l3-p5-split-s${SEED}-${SLURM_JOB_ID}" \
    --wandb_tags "curriculum,level3,10houses,chair-only,vggt,aggregator-pooled,pool-on-device,skip-heads,jax,3d-encoder" \
    --render_resolution 518 \
    --video_log_every 0 \
    --train_ratio 128 \
    --act_entropy 0.1
