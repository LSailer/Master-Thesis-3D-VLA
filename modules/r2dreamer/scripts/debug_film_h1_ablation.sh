#!/usr/bin/env bash
# Short local H100 debug runner for VGGT-FiLM H1 pose/world-point ablations.
# Usage: bash modules/r2dreamer/scripts/debug_film_h1_ablation.sh [steps] [prefill]
set -euo pipefail

STEPS="${1:-1100}"
PREFILL="${2:-1024}"
STAMP="$(date +%Y%m%d-%H%M%S)"
BASE_OUT="output/debug-film-h1-ablation/${STAMP}"
mkdir -p "${BASE_OUT}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

printf 'Debug FiLM H1 ablations on %s at %s\n' "$(hostname)" "$(date)"
printf 'GPU: '; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
printf 'steps=%s prefill=%s output=%s WANDB_MODE=%s\n' "${STEPS}" "${PREFILL}" "${BASE_OUT}" "${WANDB_MODE}"

for ABL in none zero_pose zero_wp; do
  OUT="${BASE_OUT}/${ABL}"
  mkdir -p "${OUT}"
  printf '\n=== Running vggt_film_v1 ablation=%s ===\n' "${ABL}"
  uv run python modules/r2dreamer/scripts/run_jax_habitat_vggt_film.py \
    --steps "${STEPS}" \
    --prefill "${PREFILL}" \
    --checkpoint_every "${STEPS}" \
    --output_dir "${OUT}" \
    --seed 12345 \
    --log_every 25 \
    --wandb_project 3d-vla-objectnav \
    --wandb_name "debug_film_h1_${ABL}_${STAMP}" \
    --wandb_tags "debug,h1-ablation,vggt_film_v1,${ABL}" \
    --render_resolution 518 \
    --film_ablation "${ABL}" \
    2>&1 | tee "${OUT}/console.log"

  printf 'Last FiLM/grad metrics for %s:\n' "${ABL}"
  python - <<PY
import csv, pathlib
p=pathlib.Path('${OUT}')/'metrics.csv'
keys={'grad/obs_wp_norm','grad/obs_pose_norm','grad/obs_pose_to_wp_ratio','film/gamma_minus_1_abs_mean','film/gamma_actual_mean','film/gamma_actual_std','film/beta_abs_mean','film/beta_rms'}
last={}
if p.exists():
    with p.open() as f:
        for row in csv.DictReader(f):
            if row['metric'] in keys:
                last[row['metric']] = (row['step'], row['value'])
for k in sorted(keys):
    print(f'{k}: {last.get(k, "MISSING")}')
PY
done

printf '\nDebug run complete. Outputs under %s\n' "${BASE_OUT}"
