# Slurm launcher

One YAML-backed launcher that renders and submits sbatch jobs. Replaces
per-experiment `*.sbatch` files in `scripts/r2dreamer/slurm/`.

## Quickstart

```bash
# Render only — print the sbatch script to stdout (no submission)
bash scripts/slurm/launch.sh l1_vggt --dry-run             # prod
bash scripts/slurm/launch.sh l1_vggt --smoke --dry-run     # smoke

# Submit
bash scripts/slurm/launch.sh l1_vggt --smoke               # 30-min dev job
bash scripts/slurm/launch.sh l1_vggt --prod                # 48-h full run
bash scripts/slurm/launch.sh l1_vggt --smoke-then-prod     # smoke, then prod afterok
```

The first positional argument (`l1_vggt` above) is the **variant name** —
it maps to `scripts/slurm/configs/<variant>.yaml`.

## Modes

| Mode               | Partition       | Walltime | WandB    | Purpose                        |
|--------------------|-----------------|----------|----------|--------------------------------|
| `--prod` (default) | `gpu_h100`      | 48:00:00 | online   | Real training run              |
| `--smoke`          | `dev_gpu_h100`  | 00:30:00 | offline  | Dev-cluster sanity check       |
| `--smoke-then-prod`| both            | both     | both     | Prod runs only if smoke passes |

Smoke jobs additionally:

- run with `set -euo pipefail` and `uv run --no-sync` (no resync overhead)
- assert `metrics.csv` exists with ≥5 rows; exit non-zero otherwise
- print `=== Smoke PASS ===` on success

## Config layout

```
scripts/slurm/
├── configs/
│   ├── _base.yaml          # shared sbatch directives + smoke overrides
│   └── l1_vggt.yaml        # per-variant config (extends _base)
├── launch.py               # render + validate
├── launch.sh               # CLI wrapper around sbatch
└── train.sbatch            # generic fallback sbatch (manual submit)
```

A variant config looks like:

```yaml
extends: _base
job_name: vggt_jax
output_dir: output/r2dreamer-curriculum-l1-vggt
script: scripts/r2dreamer/run_jax_habitat_vggt.py
comments:
  - "Level 1 with VGGT 3D encoder..."
args:
  steps: 2000000
  prefill: 5000
  ...
```

`extends: _base` deep-merges with `_base.yaml`. The merged dict is then
validated with pydantic (`LaunchConfig` in `launch.py`); missing or invalid
fields cause a non-zero exit **before** any `sbatch` call.

## Adding a new variant

1. Copy an existing config: `cp configs/l1_vggt.yaml configs/<name>.yaml`.
2. Edit `job_name`, `output_dir`, `script`, and any `args:` overrides.
3. Sanity check:
   ```bash
   bash scripts/slurm/launch.sh <name> --dry-run
   bash scripts/slurm/launch.sh <name> --smoke --dry-run
   ```
4. Run the smoke job:
   ```bash
   bash scripts/slurm/launch.sh <name> --smoke
   ```

## Tests

```bash
uv run pytest tests/slurm/test_launch.py -v
```

Covers:

- prod render is byte-equal to the legacy hand-written sbatch
- `--smoke-then-prod` issues the prod submit with `--dependency=afterok:<smoke_jid>`
- pydantic validation fails fast (e.g. missing `args.steps`) before any sbatch call

## Monitoring a submitted job

```bash
squeue -j <jid>                                            # status
tail -f output/<variant>/smoke/slurm-<jid>.out             # smoke logs
tail -f output/<variant>/slurm-<jid>.out                   # prod logs
sacct -j <jid> --format=JobID,State,Elapsed,MaxRSS,ExitCode
```
