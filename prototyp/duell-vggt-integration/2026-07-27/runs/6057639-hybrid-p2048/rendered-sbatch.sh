#!/bin/bash
#SBATCH --job-name=r2d-L3du-hyb
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=output/runs/duell-l3-hybrid-p2048/slurm-%j.out
#SBATCH --error=output/runs/duell-l3-hybrid-p2048/slurm-%j.err

mkdir -p output/runs/duell-l3-hybrid-p2048

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
GPU_MEMORY_LOG="output/runs/duell-l3-hybrid-p2048/gpu-memory-${SLURM_JOB_ID}.csv"
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

# Duell 2026-07-27 wave 1: thin variant of l3_hybrid for a 30-minute scored prod probe. Only prefill differs (2048 instead of the prod default 5000) to reach the replay gate (batch_size*seq_len=1024) faster and roughly double the trained steps inside the 30-minute window. See prototyp/duell-vggt-integration/2026-07-27/README.md.
uv run python scripts/r2dreamer/run.py habitat-l3-hybrid \
    --steps 1500000 \
    --prefill 2048 \
    --checkpoint_every 100000 \
    --output_dir "output/runs/duell-l3-hybrid-p2048/run-${SLURM_JOB_ID}" \
    --seed "${SEED}" \
    --log_every 250 \
    --buffer_capacity 500000 \
    --wandb_project 3d-vla-objectnav \
    --wandb_name "duell-l3-hybrid-p2048-s${SEED}-${SLURM_JOB_ID}" \
    --wandb_tags "curriculum,level3,10houses,chair-only,hybrid,cnn,wp-cp,jax,3d-encoder" \
    --render_resolution 518
