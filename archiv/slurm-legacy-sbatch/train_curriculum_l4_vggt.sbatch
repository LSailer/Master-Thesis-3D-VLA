#!/bin/bash
#SBATCH --job-name=r2d-L4-vggt
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=output/r2dreamer-curriculum-l4-vggt/slurm-%j.out
#SBATCH --error=output/r2dreamer-curriculum-l4-vggt/slurm-%j.err

mkdir -p output/r2dreamer-curriculum-l4-vggt

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"

# Generate curriculum configs if not present
if [ ! -f data/curriculum/level4_10houses_6goals.json ]; then
    echo "Generating curriculum configs..."
    uv run python scripts/environments/generate_curriculum.py
fi

# Level 4 with VGGT 3D encoder: 10 houses (4 easy, 4 medium, 2 hard) x 6 goal categories
# Full curriculum — combines multi-scene (L3) + multi-goal (L2) generalization.
# Replaces CNN with InfiniteVGGT streaming point cloud features
# 2M steps, 48h wall time (VGGT adds ~100ms/step during acting)
uv run python scripts/r2dreamer/run_jax_habitat_l4_vggt.py \
    --steps 2000000 \
    --prefill 5000 \
    --checkpoint_every 100000 \
    --output_dir "output/r2dreamer-curriculum-l4-vggt/run-${SLURM_JOB_ID}" \
    --seed "${SLURM_JOB_ID}" \
    --log_every 250 \
    --wandb_project 3d-vla-objectnav \
    --wandb_name "r2d-L4-vggt-${SLURM_JOB_ID}" \
    --wandb_tags "curriculum,level4,10houses,6goals,vggt,jax,3d-encoder" \
    --val_data data/val_replay/val_200ep.npz \
    --val_loss_every 10000 \
    --render_resolution 518
