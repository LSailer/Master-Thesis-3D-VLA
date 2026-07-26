#!/usr/bin/env bash
set -euo pipefail
# Full suite (CPU + GPU + Habitat E2E) in ONE allocation.
# Deliberate deviation from the "targeted" commands.test contract:
# this repo has no separate CI regression, so the gate owns it.
# Comma list: SLURM picks whichever partition is free first.
# --no-sync: plain `uv run` re-syncs the shared .venv, which on this cluster
# means pulling gigabytes of CUDA wheels - and on a compute node inside srun
# that is both slow and liable to fail outright.
srun --partition=dev_gpu_h100,gpu_h100_short --gres=gpu:1 --time=00:20:00 \
  bash -c 'RUN_HABITAT_E2E=1 uv run --no-sync pytest tests/ -q \
           --cov=src --cov-report=json:coverage.json'

# Coverage of changed lines feeds the risk score (measured, not blocking).
# uvx, not `uv run`: diff-cover is a reporting tool, not a project dependency.
uvx diff-cover coverage.json --compare-branch origin/main \
  --json-report .no-mistakes/diff-cover.json || true