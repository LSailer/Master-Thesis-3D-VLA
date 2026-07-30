#!/bin/bash
#SBATCH --job-name=r2d-L3d2-tr128
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=output/runs/duell2-l3-aggpool-b200k-tr128/slurm-%j.out
#SBATCH --error=output/runs/duell2-l3-aggpool-b200k-tr128/slurm-%j.err

mkdir -p output/runs/duell2-l3-aggpool-b200k-tr128

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
GPU_MEMORY_LOG="output/runs/duell2-l3-aggpool-b200k-tr128/gpu-memory-${SLURM_JOB_ID}.csv"
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

# Duell 2 (2026-07-27-r2) wave 1, slot C: the duell-1 lottery arm (kv budget 200k, prefill 1024, act_entropy 1e-1) with train_ratio halved again to 128. Rationale: in the 30-minute window metrics/sr is episode lottery, so env throughput (episodes, ms/step) dominates the scoring matrix; train_ratio 128 spends less GPU time on gradient steps per env step. Update parity: credit 128/1024 = 0.125/step, updates land on multiples of 8, log_every 250 meets them at step 1000, 2000, ... so training metrics still log.
uv run --no-sync python -m src.main \
    --env habitat \
    --steps 1500000 \
    --adapter aggregator_pooled_b200k \
    --curriculum L3 \
    --prefill 1024 \
    --checkpoint_every 100000 \
    --output_dir "output/runs/duell2-l3-aggpool-b200k-tr128/run-${SLURM_JOB_ID}" \
    --seed "${SEED}" \
    --log_every 250 \
    --buffer_capacity 500000 \
    --wandb_project 3d-vla-objectnav \
    --wandb_name "duell2-l3-aggpool-b200k-tr128-s${SEED}-${SLURM_JOB_ID}" \
    --wandb_tags "curriculum,level3,10houses,chair-only,vggt,aggregator-pooled,pool-on-device,skip-heads,jax,3d-encoder" \
    --render_resolution 518 \
    --video_log_every 0 \
    --train_ratio 128 \
    --act_entropy 0.1
