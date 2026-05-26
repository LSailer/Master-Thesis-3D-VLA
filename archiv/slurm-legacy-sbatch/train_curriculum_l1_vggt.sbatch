#!/bin/bash
#SBATCH --job-name=vggt_jax
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=output/r2dreamer-curriculum-l1-vggt/slurm-%j.out
#SBATCH --error=output/r2dreamer-curriculum-l1-vggt/slurm-%j.err

mkdir -p output/r2dreamer-curriculum-l1-vggt

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Generate curriculum configs if not present
if [ ! -f data/curriculum/level1_1house_1goal.json ]; then
    echo "Generating curriculum configs..."
    uv run python scripts/environments/generate_curriculum.py
fi

# Level 1 with VGGT 3D encoder: 1 house (fK2vEV32Lag), chair only
# Replaces CNN with InfiniteVGGT streaming point cloud features
# 2M steps, 48h wall time (VGGT adds ~100ms/step during acting)
uv run python scripts/r2dreamer/run_jax_habitat_vggt.py \
    --steps 2000000 \
    --prefill 5000 \
    --checkpoint_every 100000 \
    --output_dir "output/r2dreamer-curriculum-l1-vggt/run-${SLURM_JOB_ID}" \
    --seed "${SLURM_JOB_ID}" \
    --log_every 250 \
    --wandb_project 3d-vla-objectnav \
    --wandb_name "vggt_jax-${SLURM_JOB_ID}" \
    --wandb_tags "curriculum,level1,1house,chair-only,vggt,vggt_jax,jax,3d-encoder" \
    --render_resolution 518
