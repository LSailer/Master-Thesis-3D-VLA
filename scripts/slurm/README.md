# Slurm launcher

One YAML-backed launcher that renders and submits sbatch jobs. Replaces the
per-experiment `*.sbatch` files scattered across `scripts/` and
`scripts/r2dreamer/slurm/` with a single universal renderer plus one small
config per variant.

## Quickstart

```bash
# Render only — print the sbatch script to stdout (no submission)
bash scripts/slurm/launch.sh l1_vggt --dry-run             # prod
bash scripts/slurm/launch.sh l1_vggt --smoke --dry-run     # smoke

# Submit
bash scripts/slurm/launch.sh l1_vggt --smoke               # short dev job
bash scripts/slurm/launch.sh l1_vggt --prod                # full run
bash scripts/slurm/launch.sh l1_vggt --smoke-then-prod     # prod runs only if smoke passes

# Sweep several variants in one shell call (bash brace expansion)
bash scripts/slurm/launch.sh l{1,2,3,4}_vggt --smoke

# Override an env var (wins over the YAML default)
bash scripts/slurm/launch.sh offline_buffer_3d25 \
    --env CNN_CHECKPOINT=output/r2dreamer-curriculum-l1/run-4367942/checkpoints/step_001000000.pkl \
    --smoke
```

The first positional argument(s) are **variant names** — each maps to
`scripts/slurm/configs/<variant>.yaml`.

## Variants

| Variant              | Script                                      | Replaces (legacy sbatch)                         |
|----------------------|---------------------------------------------|--------------------------------------------------|
| `l1_vggt`            | `run_jax_habitat_vggt.py`                   | `train_curriculum_l1_vggt.sbatch`                |
| `l2_vggt`            | `run_jax_habitat_l2_vggt.py`                | `train_curriculum_l2_vggt.sbatch`                |
| `l3_vggt`            | `run_jax_habitat_l3_vggt.py`                | `train_curriculum_l3_vggt.sbatch`                |
| `l4_vggt`            | `run_jax_habitat_l4_vggt.py`                | `train_curriculum_l4_vggt.sbatch`                |
| `aggregator_mlp_v1`  | `run_jax_habitat_vggt_aggregator_mlp.py`    | `prod_aggregator_mlp_v1.sbatch` + `smoke_aggregator_mlp_fast_path.sbatch` |
| `offline_buffer_3d25`| `collect_offline_buffer.py`                 | `collect_offline_buffer_3d25.sbatch`             |

The legacy `*.sbatch` files these replace were archived in slice s5 (3D-34) under
`archiv/slurm-legacy-sbatch/` (see the README there). Non-migrated legacy scripts
(`*_actfix`, `*_rerun`, non-vggt levels, `*_resume*`, …) remain in their original
locations pending separate evaluation.

## Modes

| Mode               | Partition         | Walltime | WandB    | Purpose                          |
|--------------------|-------------------|----------|----------|----------------------------------|
| `--prod` (default) | per `sbatch.*`    | per cfg  | online   | Real training / collection run   |
| `--smoke`          | per `smoke.*`     | per cfg  | offline  | Production-faithful sanity check |
| `--smoke-then-prod`| both              | both     | both     | Prod runs only if smoke passes (`--dependency=afterok:<smoke_jid> --kill-on-invalid-dep=yes`) |

Smoke jobs additionally:

- run with `set -euo pipefail` and (for the default `uv run python`) `uv run --no-sync python`
- export `WANDB_MODE=offline`, `PYTHONFAULTHANDLER=1`,
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.7`
- when `smoke.assert_file` is set, assert that file exists in the run dir (and
  has ≥ `smoke.assert_min_rows` rows), exiting non-zero otherwise, then print
  `=== Smoke PASS ===`

> **Note on partition choice.** All habitat smokes use `gpu_h100_short`, not
> `dev_gpu_h100`. The `dev_gpu_h100` partition is a single node (`uc3n082`) where
> habitat_sim's OpenGL renderer aborts during prefill — see
> `docs/3d-30-smoke-partition-analysis.html`. `gpu_h100_short` runs on the same
> H100 class as production with a 30-min cap, so smoke validates the real path.
> This overrides the `dev_gpu_h100` choice in the original s3/s4 acceptance
> criteria, which predate that finding.

## Config schema

A variant config is validated by pydantic (`LaunchConfig` in `launch.py`)
*before* any `sbatch` call; a missing or malformed field exits non-zero with a
clear message and submits nothing.

```yaml
extends: _base            # optional; resolved recursively (a config may extend
                          # another variant, which may itself extend _base)
job_name: r2d-L2-vggt
output_dir: output/...    # directory for #SBATCH logs (the run dir lives in args)
script: scripts/r2dreamer/run_jax_habitat_l2_vggt.py   # repo-relative entrypoint

python: uv run python     # interpreter prefix; e.g. ".venv/bin/python"
arg_style: underscore     # "underscore" -> --val_data ; "hyphen" -> --val-data
strict_bash: false        # emit `set -euo pipefail` in prod too (always on for smoke)
curriculum_check: data/curriculum/level2_1house_6goals.json   # optional generate-if-missing guard

sbatch:                   # #SBATCH resource directives
  partition: gpu_h100
  gres: gpu:1
  ntasks: 1
  cpus_per_task: 8
  mem: 64G
  time: "48:00:00"

env:                      # exported before the run; overridable via --env K=V
  CNN_CHECKPOINT: output/.../step_001000000.pkl

setup:                    # raw bash lines run before the training command
  - ./scripts/setup_worktree.sh
  - bash scripts/slurm/hooks/link_external.sh

args:                     # free mapping -> `--<key> <value>` (auto-quoted, order preserved)
  steps: 2000000
  output_dir: output/.../run-${SLURM_JOB_ID}
  wandb_tags: curriculum,level2,1house,6goals

smoke:                    # mode overrides; smoke.args deep-merges onto args
  partition: gpu_h100_short
  time: "00:30:00"
  assert_file: metrics.csv
  assert_min_rows: 5
  args:
    steps: 1500
```

### How rendering works

- **`extends`** is resolved recursively (child wins; lists replace, dicts merge),
  so `l2_vggt → l1_vggt → _base` collapses into one validated config with no
  leftover `extends` key. Cycles are rejected.
- **`args`** is a free mapping. Each entry renders as a `--flag value` line in
  YAML order; flag style follows `arg_style`; values containing whitespace,
  `$`, or `,` are quoted (matching the hand-written sbatch convention).
- **`${TIMESTAMP}`** in any rendered value emits a `TIMESTAMP=$(date ...)` line.
- **`${SLURM_JOB_ID}`** and other env vars are expanded by Slurm/bash at runtime.

## Config layout

```
scripts/slurm/
├── configs/
│   ├── _base.yaml             # shared sbatch + smoke defaults (curriculum/training family)
│   ├── l1_vggt.yaml           # extends _base
│   ├── l2_vggt.yaml           # extends l1_vggt
│   ├── l3_vggt.yaml           # extends l1_vggt
│   ├── l4_vggt.yaml           # extends l1_vggt
│   ├── aggregator_mlp_v1.yaml # extends _base; timestamp run id
│   └── offline_buffer_3d25.yaml  # standalone; hyphen flags, env, setup hooks
├── hooks/
│   └── link_external.sh       # symlink external VGGT repos into a fresh worktree
├── launch.py                  # render + validate
├── launch.sh                  # CLI wrapper around sbatch (multi-variant, --env)
└── train.sbatch               # generic fallback sbatch (manual submit)
```

## Adding a new variant

1. Copy an existing config: `cp configs/l1_vggt.yaml configs/<name>.yaml`.
2. Edit `job_name`, `output_dir`, `script`, and any `args:` / `smoke:` overrides.
   Reuse a sibling via `extends:` to keep the delta tiny.
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

Covers: l1 prod render is byte-equal to the legacy sbatch; `--smoke-then-prod`
issues the prod submit with `--dependency=afterok:<smoke_jid>`; fail-fast
validation before any sbatch call; recursive `extends` (incl. cycle rejection);
L2/L3/L4 render the same training args as their legacy scripts; multi-variant
sweep submits every variant; aggregator timestamp/strict behavior; offline-buffer
hyphen flags, setup hooks, and `--env` override.

## Monitoring a submitted job

```bash
squeue -j <jid>                                            # status
tail -f output/<variant>/smoke/slurm-<jid>.out             # smoke logs
tail -f output/<variant>/slurm-<jid>.out                   # prod logs
sacct -j <jid> --format=JobID,State,Elapsed,MaxRSS,ExitCode
```
