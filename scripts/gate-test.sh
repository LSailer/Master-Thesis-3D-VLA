#!/usr/bin/env bash
set -euo pipefail
# CPU-reachable unit-test suite, run wherever it is invoked (the gate wraps
# it in a SLURM CPU allocation via scripts/gate.sh).
#
# No marker filter: conftest.py already auto-skips gpu-marked tests when no
# JAX GPU backend is visible and habitat E2E unless RUN_HABITAT_E2E=1, so a
# plain run self-documents its skips. Measured 2026-07-28: 678 passed,
# 123 skipped in 6:21 on one dev_cpu node.
#
# GPU / habitat / parity tests are NOT covered here. They are the
# responsibility of whoever touches the code they guard: run them in a GPU
# allocation (sbatch/srun) before shipping such changes.
#
# --no-sync: plain `uv run` re-syncs the shared .venv, which on this cluster
# means pulling gigabytes of CUDA wheels and can leave a stub venv behind.

cd "$(dirname "$0")/.."
uv run --no-sync pytest tests/ -q
