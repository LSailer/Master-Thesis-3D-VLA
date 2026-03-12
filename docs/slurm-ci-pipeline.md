# SLURM CI Pipeline — Dev & Prod Jobs

## Overview

Push to `dev/offline-training` triggers a GitHub Actions workflow that submits **two SLURM jobs** from `scripts/train_offline.slurm`. A short dev job validates on H100, and a long prod job runs full offline training. If dev fails, prod is automatically cancelled.

## How It Works

```
push to dev/offline-training
    │
    ▼
GitHub Actions (self-hosted runner)
    │
    ├─► sbatch --partition=dev_gpu_h100 --time=00:30:00 scripts/train_offline.slurm   (dev)
    ├─► sbatch --partition=gpu_h100_il  --time=12:00:00 scripts/train_offline.slurm   (prod)
    │
    ▼
Wait for dev job
    │
    ├─ dev passes → prod continues running
    └─ dev fails  → scancel prod job, workflow fails
```

## Single Script, Two Partitions

Both jobs use `scripts/train_offline.slurm`. The training command is defined once:

```bash
uv run src/offline/train_offline.py --config src/dreamer_config.yml
```

The slurm script has no `--time` or `--partition` directives — these are always passed via `sbatch` CLI flags. Change training params in one place, both jobs run the same thing.

## Failure Handling

1. Workflow waits for the dev job (polls `squeue` every 5 min)
2. Checks exit code via `sacct`
3. If non-zero → `scancel $PROD_ID` and `exit 1`
4. Safety net step (`if: failure()`) cancels prod on any unexpected workflow failure

## Manual Usage

```bash
# Dev (quick validation)
sbatch --partition=dev_gpu_a100_il --time=00:30:00 scripts/train_offline.slurm

# Prod (full run)
sbatch --partition=gpu_a100_il --time=48:00:00 scripts/train_offline.slurm
```
