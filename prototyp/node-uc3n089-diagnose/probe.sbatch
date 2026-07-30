#!/bin/bash
#SBATCH --job-name=node-probe
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=output/slurm/node-probe-%j.out
#SBATCH --error=output/slurm/node-probe-%j.err

# Staged diagnosis for the silent exit-134 SIGABRT on uc3n089 (see GOAL.md).
# Submit pinned to a node via submit.sh <node>. Each stage is its own python
# process with faulthandler enabled; a failing stage never stops the next, so
# one run yields a verdict per hypothesis.

set -u
REPO=/pfs/data6/home/ul/ul_student/ul_hfj15/Master-Thesis-3D-VLA
cd "$REPO"
NODE=$(hostname -s)
OUT="outputs/prototype/node-uc3n089-diagnose/${NODE}-${SLURM_JOB_ID}"
mkdir -p "$OUT" output/slurm

export PYTHONPATH="$REPO"
export PYTHONFAULTHANDLER=1
export WANDB_MODE=offline
export R2DREAMER_HARD_EXIT_ON_FINISH=1

echo "Probe on $NODE, job $SLURM_JOB_ID, $(date -Is)" | tee "$OUT/summary.txt"

# Background GPU memory sampler, same as the launcher's.
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,ecc.errors.uncorrected.volatile.total --format=csv -l 5 > "$OUT/gpu-memory.csv" 2>/dev/null &
GPU_MON=$!
trap 'kill $GPU_MON 2>/dev/null || true' EXIT

run_stage() {
    local name=$1 limit=$2; shift 2
    echo "=== stage $name: $* ===" | tee -a "$OUT/summary.txt"
    timeout -k 30 "$limit" "$@" > "$OUT/$name.out" 2> "$OUT/$name.err"
    local rc=$?
    echo "stage $name rc=$rc $( [ $rc -eq 124 ] && echo '(TIMEOUT)' )" | tee -a "$OUT/summary.txt"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader >> "$OUT/summary.txt" 2>/dev/null
}

# Stage 0 - hardware state: ECC counters, remapped rows, driver, clocks.
nvidia-smi -q > "$OUT/nvidia-smi-q.txt" 2>&1
nvidia-smi -q -d ECC,ROW_REMAPPER,MEMORY,CLOCK > "$OUT/nvidia-smi-focus.txt" 2>&1
grep -A4 "Remapped Rows" "$OUT/nvidia-smi-focus.txt" | tee -a "$OUT/summary.txt" || true

# Stage 1 - H1 GL only: habitat prefill with JAX forced onto the CPU.
JAX_PLATFORMS=cpu run_stage habitat-cpu 600 \
    uv run --no-sync python -m src.main \
    --adapter rgb --curriculum L1 --steps 1 --prefill 300 \
    --batch_size 4 --seq_len 16 --train_ratio 1 --log_every 50 \
    --checkpoint_every 1000000 --seed 42 --video_log_every 0 \
    --wandb_project node-probe --wandb_name "probe-hab-cpu-$NODE-$SLURM_JOB_ID" \
    --output_dir "$OUT/habitat-cpu"

# Stage 2 - H2 XLA only: default preallocation plus sustained matmuls, no GL.
run_stage jax-gpu 300 \
    uv run --no-sync python -c "
import jax, jax.numpy as jnp
print(jax.devices())
x = jnp.ones((8192, 8192), jnp.bfloat16)
for i in range(200):
    x = (x @ x) / 8192.0
x.block_until_ready()
print('matmuls ok', jax.devices()[0].memory_stats())
"

# Stage 3 - H3 prod ordering: XLA prealloc first, then habitat GL, prod shape.
run_stage combined 900 \
    uv run --no-sync python -m src.main \
    --adapter rgb --curriculum L1 --steps 100 --prefill 2048 \
    --batch_size 16 --seq_len 64 --train_ratio 512 --log_every 10 \
    --checkpoint_every 1000000 --seed 42 --video_log_every 0 \
    --wandb_project node-probe --wandb_name "probe-combined-$NODE-$SLURM_JOB_ID" \
    --output_dir "$OUT/combined"

echo "=== done $(date -Is) ===" | tee -a "$OUT/summary.txt"
cat "$OUT/summary.txt"
