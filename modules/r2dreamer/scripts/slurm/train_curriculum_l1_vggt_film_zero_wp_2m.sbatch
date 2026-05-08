#!/bin/bash
#SBATCH --job-name=film_zwp_2m
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=output/r2dreamer-curriculum-l1-vggt-film-zero-wp/slurm-%j.out
#SBATCH --error=output/r2dreamer-curriculum-l1-vggt-film-zero-wp/slurm-%j.err

set -euo pipefail

mkdir -p output/r2dreamer-curriculum-l1-vggt-film-zero-wp

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Run: VGGT-FiLM v1, L1, 2M steps, zero_wp ablation"
echo "Git: $(git rev-parse --short HEAD) branch $(git branch --show-current) dirty=$(test -n "$(git status --short)" && echo yes || echo no)"

if [ ! -f data/curriculum/level1_1house_1goal.json ]; then
    echo "Generating curriculum configs..."
    uv run python modules/envs/scripts/generate_curriculum.py
fi

uv run python modules/r2dreamer/scripts/run_jax_habitat_vggt_film.py \
    --steps 2000000 \
    --prefill 5000 \
    --checkpoint_every 100000 \
    --output_dir "output/r2dreamer-curriculum-l1-vggt-film-zero-wp/run-${SLURM_JOB_ID}" \
    --seed "${SLURM_JOB_ID}" \
    --log_every 250 \
    --wandb_project 3d-vla-objectnav \
    --wandb_name "film_vggt_v1_zero_wp_2m-${SLURM_JOB_ID}" \
    --wandb_tags "curriculum,level1,1house,chair-only,vggt,vggt_film_v1,film,zero_wp,jax,3d-encoder,2m,48h,h1-ablation,diagnostic-metrics" \
    --render_resolution 518 \
    --film_ablation zero_wp
