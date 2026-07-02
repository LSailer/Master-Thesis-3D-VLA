#!/bin/bash
#SBATCH --job-name=r2d-graph-house
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=output/bench/r2d-graph-house-%j.out
#SBATCH --error=output/bench/r2d-graph-house-%j.err

set -euo pipefail

# Graph-spectral house-context experiments (src/prototyp/graph_house_context):
# k-NN graph construction, contour-aware downsampling vs stride, block-GFT RGB
# compression, and the sparse GCN RGB demo — all on the saved 1 cm house
# cloud. GPU accelerates the jaxkd k-NN queries and the GCN training; exp3's
# per-block eigh loop is host-bound either way.

mkdir -p output/bench

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Branch: $(git branch --show-current)"
echo "Commit: $(git rev-parse --short HEAD)"
echo "Dirty: $(git status --porcelain | wc -l) files"

export PYTHONFAULTHANDLER=1

uv run python -m src.prototyp.graph_house_context.exp1_build_graph --cuda
uv run python -m src.prototyp.graph_house_context.exp2_contour_downsample --cuda
uv run python -m src.prototyp.graph_house_context.exp3_gft_compress
uv run python -m src.prototyp.graph_house_context.exp4_gcn_rgb_demo --cuda --steps 2000

echo "=== Graph house-context experiments done ==="
