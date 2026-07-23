#!/bin/bash
#SBATCH --job-name=r2d-persist-align
#SBATCH --partition=gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=output/check/persist-align-%j.out
#SBATCH --error=output/check/persist-align-%j.err

# Run the PERSIST-vs-FULL cross-episode alignment diagnostic
# (src/prototyp/live_house_context/check_persist_alignment.py). Short GPU job:
# 3 episodes x 60 steps under each mode (~6-10 min). Pass = PERSIST %new << FULL
# %new and PERSIST IoU@10cm >> FULL IoU@10cm. See PROTOCOL.md.

set -euo pipefail
mkdir -p output/check

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Commit: $(git rev-parse --short HEAD)"

scripts/slurm/hooks/link_external.sh

if [ ! -f data/curriculum/level1_1house_1goal.json ]; then
    echo "Generating curriculum configs..."
    uv run python scripts/environments/generate_curriculum.py
fi

uv run python src/prototyp/live_house_context/check_persist_alignment.py \
    --total 80 --split 60 --tail 20

echo "=== diagnostic done ==="