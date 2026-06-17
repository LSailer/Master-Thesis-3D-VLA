#!/bin/bash
#SBATCH --job-name=3d48-wp-l1
#SBATCH --partition=dev_gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=output/3d48/wp-l1-%j.log
#SBATCH --error=output/3d48/wp-l1-%j.log

set -euo pipefail
REPO="/pfs/data6/home/ul/ul_student/ul_hfj15/Master-Thesis-3D-VLA"
cd "$REPO"
mkdir -p output/3d48

echo "=== 3D-48 dense-vs-pooled dump on real L1 frame (episode_009/step_038) ==="
echo "Node: $(hostname)"; echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"; echo "Date: $(date)"

# Use the venv python directly (not `uv run`): matches the working VGGT
# training jobs and lets jaxlib resolve cuDNN via the installed
# nvidia-cudnn-cu12 package instead of re-syncing the environment.
.venv/bin/python -m scripts.profiling.diagnostics.dump_wp_dense_vs_pooled \
    --frames output/3d48/frame038_input.npz \
    --frame-index 0 \
    --out output/3d48/wp_dense_vs_pooled.npz

echo "=== Done at $(date) ==="
