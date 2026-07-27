#!/bin/bash
#SBATCH --job-name=r2d-L3du-agglo
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=output/runs/duell-l3-aggpool-lottery/slurm-%j.out
#SBATCH --error=output/runs/duell-l3-aggpool-lottery/slurm-%j.err

mkdir -p output/runs/duell-l3-aggpool-lottery

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
GPU_MEMORY_LOG="output/runs/duell-l3-aggpool-lottery/gpu-memory-${SLURM_JOB_ID}.csv"
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

# Duell 2026-07-27 wave 2, aggressive arm: pooled tokens with the KV cache capped at 200k slots (adapter aggregator_pooled_b200k, src/adapters/global_tokens.py) plus the three lottery knobs from the danijar analysis (prefill 1024, train_ratio 256, act_entropy 1e-1). The budget cap cuts the per-step eviction top_k cost; the jianyuan analysis (prototyp/duell-vggt-integration/2026-07-27/agents/jianyuan-wang/NOTES.md) estimates -20 to -30 ms/step at the price of a ~6-frame context window.
uv run python scripts/r2dreamer/run.py habitat-l3-aggregator-pooled-b200k \
    --steps 1500000 \
    --prefill 1024 \
    --checkpoint_every 100000 \
    --output_dir "output/runs/duell-l3-aggpool-lottery/run-${SLURM_JOB_ID}" \
    --seed "${SEED}" \
    --log_every 250 \
    --buffer_capacity 500000 \
    --wandb_project 3d-vla-objectnav \
    --wandb_name "duell-l3-aggpool-lottery-s${SEED}-${SLURM_JOB_ID}" \
    --wandb_tags "curriculum,level3,10houses,chair-only,vggt,aggregator-pooled,pool-on-device,skip-heads,jax,3d-encoder" \
    --render_resolution 518 \
    --video_log_every 0 \
    --val_every 0 \
    --train_ratio 256 \
    --act_entropy 0.1
