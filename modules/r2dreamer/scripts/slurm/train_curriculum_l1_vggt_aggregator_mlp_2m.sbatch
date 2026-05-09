#!/bin/bash
#SBATCH --job-name=vggt_agg_mlp
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=output/r2dreamer-curriculum-l1-vggt-aggregator-mlp/slurm-%j.out
#SBATCH --error=output/r2dreamer-curriculum-l1-vggt-aggregator-mlp/slurm-%j.err

set -euo pipefail

mkdir -p output/r2dreamer-curriculum-l1-vggt-aggregator-mlp

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Git commit: $(git rev-parse HEAD)"
echo "Git branch: $(git branch --show-current)"

# Generate curriculum configs if not present
if [ ! -f data/curriculum/level1_1house_1goal.json ]; then
    echo "Generating curriculum configs..."
    uv run --python 3.11 python modules/envs/scripts/generate_curriculum.py
fi

uv run --python 3.11 python modules/r2dreamer/scripts/run_jax_habitat_vggt_aggregator_mlp.py \
    --steps 2000000 \
    --prefill 5000 \
    --checkpoint_every 100000 \
    --output_dir "output/r2dreamer-curriculum-l1-vggt-aggregator-mlp/run-${SLURM_JOB_ID}" \
    --seed "${SLURM_JOB_ID}" \
    --log_every 250 \
    --wandb_project 3d-vla-objectnav \
    --wandb_name "variant-1-aggregator-mlp-${SLURM_JOB_ID}" \
    --wandb_tags "curriculum,level1,1house,chair-only,vggt,aggregator-mlp,variant-1,jax,3d-encoder" \
    --wandb_notes_file LEARNING.md \
    --render_resolution 518
