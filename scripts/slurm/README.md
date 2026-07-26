# Slurm launcher

One YAML-backed launcher that renders and submits sbatch jobs. Replaces the
per-experiment `*.sbatch` files scattered across `scripts/` and
`scripts/r2dreamer/slurm/` with a single universal renderer plus one small
config per variant.

## Quickstart

```bash
# Render only — print the sbatch script to stdout (no submission)
bash scripts/slurm/launch.sh l1_cnn_cap1m --dry-run             # prod
bash scripts/slurm/launch.sh l1_cnn_cap1m --smoke --dry-run     # smoke

# Submit
bash scripts/slurm/launch.sh l1_cnn_cap1m --smoke               # short dev job
bash scripts/slurm/launch.sh l1_cnn_cap1m --prod                # full run
bash scripts/slurm/launch.sh l1_cnn_cap1m --smoke-then-prod     # prod runs only if smoke passes
bash scripts/slurm/launch.sh l1_cnn_cap1m --time 04:00:00       # prod-shaped profiling run
bash scripts/slurm/launch.sh house_context_l1_long_smoke --smoke      # long post-warm smoke
bash scripts/slurm/launch.sh hybrid_house_points_pose_l1_live --smoke  # live voxel house map

# Sweep several variants in one shell call (bash brace expansion)
bash scripts/slurm/launch.sh l1_cnn_cap{10k,100k,500k,1m} --smoke

# Override an env var (wins over the YAML default)
bash scripts/slurm/launch.sh l1_cnn_cap1m \
    --env WANDB_MODE=offline \
    --smoke
```

The first positional argument(s) are **variant names** — each maps to
`scripts/slurm/configs/<variant>.yaml`.

## Variants

Curriculum-family variants share the single `scripts/r2dreamer/run.py` dispatcher
(the `script` is inherited from `_base`); each selects its run via `run_id:`, and
that run id resolves to one adapter key in `src.adapters.ADAPTERS`. There is no
encoder-type string anywhere in this layer — the adapter declares its own render
resolution, extractor settings and branch routing, so a config never has to
restate a shape.

| Variant                                 | Run id (`run_id:`)                           | Adapter                    |
|-----------------------------------------|----------------------------------------------|----------------------------|
| `l1_cnn_cap10k`                         | `habitat-l1-cnn`                             | `rgb`                      |
| `l1_cnn_cap100k`                        | `habitat-l1-cnn`                             | `rgb`                      |
| `l1_cnn_cap500k`                        | `habitat-l1-cnn`                             | `rgb`                      |
| `l1_cnn_cap1m`                          | `habitat-l1-cnn`                             | `rgb`                      |
| `l1_cnn_cap1m_seed42_fp32_probe`        | `habitat-l1-cnn`                             | `rgb`                      |
| `l1_cnn_cap1m_seed42_bf16_probe`        | `habitat-l1-cnn`                             | `rgb`                      |
| `l1_cnn_cap1m_seed42_fp16_probe`        | `habitat-l1-cnn`                             | `rgb`                      |
| `l2_cnn`                                | `habitat-l2-cnn`                             | `rgb`                      |
| `l3_cnn`                                | `habitat-l3-cnn`                             | `rgb`                      |
| `l4_cnn`                                | `habitat-l4-cnn`                             | `rgb`                      |
| `hybrid_v1`                             | `habitat-l1-hybrid`                          | `rgb_pointmap_pose`        |
| `pointmap_pose_l1`                      | `habitat-l1-pointmap-pose`                   | `pointmap_pose`            |
| `pointmap_pose_64_l1`                   | `habitat-l1-pointmap-pose-64`                | `pointmap_pose_64`         |
| `pointmap_dense_l1`                     | `habitat-l1-pointmap-dense`                  | `pointmap_dense`           |
| `hybrid_house_points_pose_l1_live`      | `habitat-l1-vggt-hybrid-house-points-pose`   | `rgb_house_voxels`         |
| `hybrid_hpp_prodshape_probe`            | `habitat-l1-vggt-hybrid-house-points-pose`   | `rgb_house_voxels`         |
| `hybrid_hpp_bf16_prodshape_probe`       | `habitat-l1-vggt-hybrid-house-points-pose`   | `rgb_house_voxels`         |
| `jax_buffer_stepopt_probe`              | `habitat-l1-vggt-hybrid-house-points-pose`   | `rgb_house_voxels`         |
| `gnn_house_points_pose_l1_live`         | `habitat-l1-gnn-house-points-pose`           | `rgb_house_voxels_gnn`     |
| `gnn_house_points_pose_l1_live_plydump` | `habitat-l1-gnn-house-points-pose`           | `rgb_house_voxels_gnn`     |
| `house_context_l1`                      | `habitat-l1-vggt-house-context`              | `rgb_house_cloud_episodes` |
| `house_context_l1_long_smoke`           | `habitat-l1-vggt-house-context`              | `rgb_house_cloud_episodes` |
| `global_tokens_l1`                      | `habitat-l1-vggt-house-global-tokens-nogate` | `rgb_global_tokens`        |
| `full_tokens_l1`                        | `habitat-l1-full-tokens`                     | `rgb_full_tokens`          |
| `aggregator_pooled_l1`                  | `habitat-l1-aggregator-pooled`               | `aggregator_pooled`        |
| `house_points_pose_l1_live`             | *(none — abstract parent)*                   | —                          |

Only the `rgb` baseline has the full L1-L4 ladder; every other arm has an L1
config only. Replay-row size is the reason the token and dense arms cap
`buffer_capacity` in both prod and smoke: a row is 5.6 MB (`rgb_full_tokens`),
2.8 MB (`rgb_global_tokens`) or 1.6 MB (`pointmap_dense`), and the buffer
preallocates `capacity` rows. The `_base` smoke block only overrides prod args,
so a cap that is missing from the smoke block is not inherited from anywhere.

`house_points_pose_l1_live` sets `run_id: null` on purpose. Its own arm (the
map-as-replacement `habitat-l1-vggt-house-points-pose` run) was dropped in the
adapter-routing migration, but the two live arms above still `extends:` it for
their shared sbatch and smoke shape. Rendering it produces a `run.py` command
with no run id, so a stray submit fails fast instead of silently training the
wrong arm. The explicit `null` matters: omitting the key would inherit
`house_context_l1`'s run id.

Standalone profiling variants use `scripts/profiling/*` entrypoints and write
Slurm logs/artifacts under `output/profiling/`: `profile_encoder_cost`,
`profile_agg_pipeline`, `profile_house_points_pose`, `profile_modal_replay`.

> **Stale.** All four point at `scripts/profiling/*.py` entrypoints that no
> longer exist (removed well before the adapter migration). They still render,
> but they will not run. Restore the scripts from git history or delete the
> configs before relying on them.

The legacy `*.sbatch` files this launcher replaced were archived in slice s5
(3D-34) and have since been deleted; recover them from git history if needed.
The remaining hand-written `scripts/r2dreamer/slurm/*.sbatch` files
(`*_actfix`, `*_rerun`, `train_habitat_baseline`, the GNN 50k comparisons) all
target surviving CNN/GNN run ids and are kept only for reproducing their
historical jobs — new work goes through this launcher.

## Modes

| Mode               | Partition         | Walltime | WandB    | Purpose                          |
|--------------------|-------------------|----------|----------|----------------------------------|
| `--prod` (default) | per `sbatch.*`    | per cfg  | online   | Real training / collection run   |
| `--smoke`          | per `smoke.*`     | per cfg  | offline  | Production-faithful sanity check |
| `--smoke-then-prod`| both              | both     | both     | Prod runs only if smoke passes (`--dependency=afterok:<smoke_jid> --kill-on-invalid-dep=yes`) |

Use `--time <SLURM_TIME>` to override the rendered `#SBATCH --time` without
creating a temporary YAML config. This is useful for prod-shaped profiling runs,
for example `bash scripts/slurm/launch.sh house_context_l1 --time 04:00:00`.
With `--smoke-then-prod`, the override applies to both submitted jobs.

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
job_name: r2d-L4-cnn
output_dir: output/...    # directory for #SBATCH logs (the run dir lives in args)
script: scripts/r2dreamer/run.py   # repo-relative entrypoint (usually inherited from _base)
run_id: habitat-l4-cnn    # leading positional for the run.py dispatcher (a RUN_CONFIGS key,
                          # which in turn names one src.adapters.ADAPTERS key).
                          # `run_id: null` blanks an inherited value, making the
                          # config an abstract parent that cannot be launched.

python: uv run python     # interpreter prefix; e.g. ".venv/bin/python"
arg_style: underscore     # "underscore" -> --val_data ; "hyphen" -> --val-data
strict_bash: false        # emit `set -euo pipefail` in prod too (always on for smoke)
curriculum_check: data/curriculum/level4_10houses_6goals.json   # optional generate-if-missing guard

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
  wandb_tags: curriculum,level4,10houses,6goals

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
  so `gnn_house_points_pose_l1_live → house_points_pose_l1_live →
  house_context_l1 → _base` collapses into one validated config with no leftover
  `extends` key. Cycles are rejected.
- **`args`** is a free mapping. Each entry renders as a `--flag value` line in
  YAML order; flag style follows `arg_style`; values containing whitespace,
  `$`, or `,` are quoted (matching the hand-written sbatch convention).
- **`${TIMESTAMP}`** in any rendered value emits a `TIMESTAMP=$(date ...)` line.
- **`${SLURM_JOB_ID}`** and other env vars are expanded by Slurm/bash at runtime.

## Config layout

```
scripts/slurm/
├── configs/
│   ├── _base.yaml                  # shared sbatch + smoke defaults (curriculum/training family)
│   ├── _profiling_base.yaml        # shared sbatch + smoke defaults (standalone profilers)
│   ├── l1_cnn_cap1m.yaml           # extends _base
│   ├── house_context_l1.yaml       # extends _base
│   ├── house_points_pose_l1_live.yaml   # extends house_context_l1; run_id: null (abstract)
│   ├── gnn_house_points_pose_l1_live.yaml     # extends house_points_pose_l1_live
│   └── hybrid_house_points_pose_l1_live.yaml  # extends house_points_pose_l1_live
├── hooks/
│   └── link_external.sh       # symlink external VGGT repos into a fresh worktree
├── launch.py                  # render + validate
├── launch.sh                  # CLI wrapper around sbatch (multi-variant, --env)
└── train.sbatch               # generic fallback sbatch (manual submit)
```

## Adding a new variant

A new *observation* variant is an adapter plus one `ADAPTERS` row plus one
`RUN_CONFIGS` row — see `scripts/r2dreamer/AGENTS.md`. A new config here is only a
new *job shape* (resources, step budget, smoke gate) over an existing run id.

1. Copy an existing config: `cp configs/l1_cnn_cap1m.yaml configs/<name>.yaml`.
2. Edit `job_name`, `output_dir`, `run_id`, and any `args:` / `smoke:` overrides.
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

Covers: `--smoke-then-prod` issues the prod submit with
`--dependency=afterok:<smoke_jid>`; fail-fast validation before any sbatch call;
recursive `extends` (incl. cycle rejection); multi-variant sweep submits every
variant; strict-bash and timestamp behavior; and `--env` / `--time` override
handling.

> **Stale.** The suite still parametrizes over configs that the adapter-routing
> migration deleted (`l1_vggt`, `l2_vggt`..`l4_vggt`, `aggregator_mlp_v1`,
> `l1_agg_mlp_cap500k`, `l1_vggt_wpcp{37,64}_cap500k`,
> `house_full_tokens_nogate*`, `house_global_embedding_l1`,
> `gnn_edge_house_points_pose_l1_live`, `profile_training_vggt`), so those cases
> fail with `FileNotFoundError` until they are repointed at surviving variants.

## Monitoring a submitted job

```bash
squeue -j <jid>                                            # status
tail -f output/<variant>/smoke/slurm-<jid>.out             # smoke logs
tail -f output/<variant>/slurm-<jid>.out                   # prod logs
sacct -j <jid> --format=JobID,State,Elapsed,MaxRSS,ExitCode
```
