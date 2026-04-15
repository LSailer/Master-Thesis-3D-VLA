---
name: experiment
description: Generate a SLURM .sbatch script from existing templates and submit with sbatch. Use after /engineer completes a phase and the code is ready to run on the cluster.
disable-model-invocation: true
---

# Experiment

Generate and submit a SLURM experiment.

## Inputs

`$ARGUMENTS` — experiment name and optional overrides, e.g.:
- `/experiment l2-vggt-curriculum`
- `/experiment l1-rerun --partition gpu_h100_il --time 24:00:00`

## Current context

- Branch: !`git branch --show-current`
- Last commit: !`git log --oneline -1`
- Uncommitted changes: !`git status --short`

## Process

### 1. Find the best template

Search existing `.sbatch` files for the closest match to the experiment description:

```bash
find modules/ scripts/ -name '*.sbatch' -type f
```

Read the top 2-3 candidates and select the one whose training script, config, and resource requests best match the requested experiment.

### 2. Check for a wiki experiment page

Look in `docs/wiki/experiments/` for a page matching the experiment name. If one exists, read it for context (setup, hypothesis, config). If none exists, note that `/engineer` should have created one — warn the user.

### 3. Generate the .sbatch script

Create a new `.sbatch` file following these conventions from the codebase:

**SBATCH header** (standard across all scripts):
```bash
#!/bin/bash
#SBATCH --job-name=<short-name>
#SBATCH --partition=<partition>       # default: gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=<time>                 # default: 24:00:00
#SBATCH --output=output/<experiment>/slurm-%j.out
#SBATCH --error=output/<experiment>/slurm-%j.err
```

**Preamble** (standard):
```bash
mkdir -p output/<experiment>

echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
```

**Training command** (adapted from template):
```bash
uv run python <training_script> \
    --steps <steps> \
    --prefill 5000 \
    --checkpoint_every 100000 \
    --output_dir "output/<experiment>/run-${SLURM_JOB_ID}" \
    --seed "${SLURM_JOB_ID}" \
    --log_every 250 \
    --wandb_project 3d-vla-objectnav \
    --wandb_name "<experiment>-${SLURM_JOB_ID}" \
    --wandb_tags "<comma-separated-tags>" \
    <additional flags from template>
```

### 4. Place the script

Write to `modules/<module>/scripts/slurm/<experiment>.sbatch` — matching the pattern of existing scripts.

### 5. Show and confirm

Display the full script to the user. Ask for confirmation before submitting. If the user has uncommitted changes, warn them.

### 6. Submit

```bash
sbatch modules/<module>/scripts/slurm/<experiment>.sbatch
```

Report the job ID. Suggest monitoring with:
```bash
squeue -j <job_id>
tail -f output/<experiment>/slurm-<job_id>.out
```

## Partition reference

| Partition | Use Case | Max Time | GPUs |
|-----------|----------|----------|------|
| `dev_gpu_h100` | Testing, validation, TDD GPU tests | 30 min | 1-4 H100 |
| `gpu_h100` | Standard GPU training | 48h | 1-4 H100 |
| `gpu_h100_il` | Production training (interactive-like) | 24h | 1-4 H100 |

## Rules

- **Never submit without user confirmation** — GPU hours cost real money
- **Warn if uncommitted changes** — the cluster runs whatever is on disk, not what's committed
- **Use SLURM_JOB_ID as seed** — reproducibility convention from existing scripts
- **W&B project is always `3d-vla-objectnav`** — don't change this
- **Default to 2.4M steps** unless the user specifies otherwise (standard from existing experiments)
