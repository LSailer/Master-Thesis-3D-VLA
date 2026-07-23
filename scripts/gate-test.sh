#!/usr/bin/env bash
set -euo pipefail
# Full suite (CPU + GPU + Habitat E2E) in ONE allocation.
# Deliberate deviation from the "targeted" commands.test contract:
# this repo has no separate CI regression, so the gate owns it.
# Comma list: SLURM picks whichever partition is free first.
srun --partition=dev_gpu_h100,gpu_h100_short --gres=gpu:1 --time=00:20:00 \
  bash -c 'RUN_HABITAT_E2E=1 uv run pytest tests/ -q \
           --cov=src --cov-report=json:coverage.json'

# Coverage of changed lines feeds the risk score (measured, not blocking)
uv run diff-cover coverage.json --compare-branch origin/main \
  --json-report .no-mistakes/diff-cover.json || true