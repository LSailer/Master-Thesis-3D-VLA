#!/usr/bin/env bash
set -euo pipefail

# Run the GPU-only decoder overfit probe through SLURM.
# Optional overrides:
#   PARTITION=gpu_h100_short TIME=00:20:00 GRES=gpu:1 ./scripts/r2dreamer/run_decoder_probe_overfit_gpu.sh -vv

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PARTITION="${PARTITION:-dev_gpu_h100}"
GRES="${GRES:-gpu:1}"
TIME="${TIME:-00:20:00}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
PYTEST_ARGS=(
  "tests/r2dreamer/world_model/test_decoder_probe_overfit_gpu.py"
  "-q"
)

cd "$ROOT"
exec srun \
  --partition="$PARTITION" \
  --gres="$GRES" \
  --time="$TIME" \
  --cpus-per-task="$CPUS_PER_TASK" \
  uv run pytest "${PYTEST_ARGS[@]}" "$@"
