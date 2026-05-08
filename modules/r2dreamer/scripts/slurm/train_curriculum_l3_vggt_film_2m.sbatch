#!/bin/bash
#SBATCH --job-name=film_l3_2m
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=output/r2dreamer-curriculum-l3-vggt-film/slurm-%j.out
#SBATCH --error=output/r2dreamer-curriculum-l3-vggt-film/slurm-%j.err

set -euo pipefail

mkdir -p output/r2dreamer-curriculum-l3-vggt-film

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Run: VGGT-FiLM v1, L3 (10 houses, chair-only), 2M steps, diagnostic metrics"
echo "Git: $(git rev-parse --short HEAD) branch $(git branch --show-current) dirty=$(test -n "$(git status --short)" && echo yes || echo no)"

if [ ! -f data/curriculum/level3_10houses_1goal.json ]; then
    echo "Generating curriculum configs..."
    uv run python modules/envs/scripts/generate_curriculum.py
fi

uv run python modules/r2dreamer/scripts/run_jax_habitat_vggt_film_l3.py \
    --steps 2000000 \
    --prefill 5000 \
    --checkpoint_every 100000 \
    --output_dir "output/r2dreamer-curriculum-l3-vggt-film/run-${SLURM_JOB_ID}" \
    --seed "${SLURM_JOB_ID}" \
    --log_every 250 \
    --wandb_project 3d-vla-objectnav \
    --wandb_name "film_vggt_v1_l3_2m-${SLURM_JOB_ID}" \
    --wandb_tags "curriculum,level3,10houses,chair-only,vggt,vggt_film_v1,film,jax,3d-encoder,2m,48h,diagnostic-metrics" \
    --render_resolution 518 \
    --val_data data/val_replay/val_200ep.npz \
    --val_loss_every 100000
