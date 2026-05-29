#!/bin/bash
#SBATCH --job-name=3d48-wp-dump
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=output/3d48/wp-dump-%j.log
#SBATCH --error=output/3d48/wp-dump-%j.log

set -euo pipefail

REPO="/pfs/data6/home/ul/ul_student/ul_hfj15/Master-Thesis-3D-VLA"
cd "$REPO"
mkdir -p output/3d48

echo "=== 3D-48 dense-vs-pooled world-point dump ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"
echo ""

uv run python -m scripts.vggt.dump_wp_dense_vs_pooled \
    --frames tests/r2dreamer/launch/fixtures/sample_habitat_obs.npz \
    --frame-index 0 \
    --out output/3d48/wp_dense_vs_pooled.npz

echo ""
echo "=== Done at $(date) ==="
