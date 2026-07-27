#!/bin/bash
#SBATCH --job-name=r2d-L3du-hyblo
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=output/runs/duell-l3-hybrid-lottery/slurm-%j.out
#SBATCH --error=output/runs/duell-l3-hybrid-lottery/slurm-%j.err

mkdir -p output/runs/duell-l3-hybrid-lottery

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
GPU_MEMORY_LOG="output/runs/duell-l3-hybrid-lottery/gpu-memory-${SLURM_JOB_ID}.csv"
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

# Duell 2026-07-27 wave 2, conservative arm: l3_hybrid with the three lottery knobs from the danijar analysis (prototyp/duell-vggt-integration/2026-07-27/agents/danijar-hafner/NOTES.md). prefill 1024 hits the replay gate exactly (batch 16 * seq 64, loops.py), train_ratio 256 halves the training share of each step for ~24% more env steps and episodes, act_entropy 1e-1 keeps the policy from collapsing below random in the ~20-episode window a 30-minute run allows.
uv run python scripts/r2dreamer/run.py habitat-l3-hybrid \
    --steps 1500000 \
    --prefill 1024 \
    --checkpoint_every 100000 \
    --output_dir "output/runs/duell-l3-hybrid-lottery/run-${SLURM_JOB_ID}" \
    --seed "${SEED}" \
    --log_every 250 \
    --buffer_capacity 500000 \
    --wandb_project 3d-vla-objectnav \
    --wandb_name "duell-l3-hybrid-lottery-s${SEED}-${SLURM_JOB_ID}" \
    --wandb_tags "curriculum,level3,10houses,chair-only,hybrid,cnn,wp-cp,jax,3d-encoder" \
    --render_resolution 518 \
    --train_ratio 256 \
    --act_entropy 0.1
