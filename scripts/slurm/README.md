# Slurm launcher

One YAML-backed launcher that renders and submits sbatch jobs. Replaces the
per-experiment `*.sbatch` files scattered across `scripts/` and
`scripts/r2dreamer/slurm/` with a single universal renderer plus one small
config per variant.

## Quickstart

```bash
# Render only - print the sbatch script to stdout (no submission)
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

The first positional argument(s) are **variant names**: each maps to
`scripts/slurm/configs/<variant>.yaml`.

## Variants

Every variant launches the single `python -m src.main` entry point (the `script`
is inherited from `_base`) and names its arm with ordinary flags in `args:`:
`env` (from `_base`), `adapter` - one key of `src.adapters.ADAPTERS` - and
`curriculum`. There is no dispatcher positional and no encoder-type string
anywhere in this layer: the adapter declares its own render resolution,
extractor settings and branch routing, so a config never has to restate a shape.

| Variant                                 | Adapter (`adapter:`)       | Curriculum |
|-----------------------------------------|----------------------------|------------|
| `l1_cnn`                                | `rgb`                      | `L1`       |
| `l1_cnn_cap10k`                         | `rgb`                      | `L1`       |
| `l1_cnn_cap100k`                        | `rgb`                      | `L1`       |
| `l1_cnn_cap500k`                        | `rgb`                      | `L1`       |
| `l1_cnn_cap1m`                          | `rgb`                      | `L1`       |
| `l1_cnn_cap1m_seed42_fp32_probe`        | `rgb`                      | `L1`       |
| `l1_cnn_cap1m_seed42_bf16_probe`        | `rgb`                      | `L1`       |
| `l1_cnn_cap1m_seed42_fp16_probe`        | `rgb`                      | `L1`       |
| `l2_cnn`                                | `rgb`                      | `L2`       |
| `l3_cnn`                                | `rgb`                      | `L3`       |
| `l4_cnn`                                | `rgb`                      | `L4`       |
| `hybrid_v1`                             | `rgb_pointmap_pose`        | `L1`       |
| `l2_hybrid`                             | `rgb_pointmap_pose`        | `L2`       |
| `l3_hybrid`                             | `rgb_pointmap_pose`        | `L3`       |
| `l4_hybrid`                             | `rgb_pointmap_pose`        | `L4`       |
| `duell_l3_hybrid_p2048`                 | `rgb_pointmap_pose`        | `L3`       |
| `duell_l3_hybrid_lottery`               | `rgb_pointmap_pose`        | `L3`       |
| `pointmap_pose_l1`                      | `pointmap_pose`            | `L1`       |
| `l2_pointmap_pose`                      | `pointmap_pose`            | `L2`       |
| `l3_pointmap_pose`                      | `pointmap_pose`            | `L3`       |
| `l4_pointmap_pose`                      | `pointmap_pose`            | `L4`       |
| `duell2_l3_pointmap_p2048`              | `pointmap_pose`            | `L3`       |
| `pointmap_pose_64_l1`                   | `pointmap_pose_64`         | `L1`       |
| `pointmap_dense_l1`                     | `pointmap_dense`           | `L1`       |
| `hybrid_house_points_pose_l1_live`      | `rgb_house_voxels`         | `L1`       |
| `hybrid_hpp_prodshape_probe`            | `rgb_house_voxels`         | `L1`       |
| `hybrid_hpp_bf16_prodshape_probe`       | `rgb_house_voxels`         | `L1`       |
| `jax_buffer_stepopt_probe`              | `rgb_house_voxels`         | `L1`       |
| `gnn_house_points_pose_l1_live`         | `rgb_house_voxels_gnn`     | `L1`       |
| `gnn_house_points_pose_l1_live_plydump` | `rgb_house_voxels_gnn`     | `L1`       |
| `house_context_l1`                      | `rgb_house_cloud_episodes` | `L1`       |
| `house_context_l1_long_smoke`           | `rgb_house_cloud_episodes` | `L1`       |
| `global_tokens_l1`                      | `rgb_global_tokens`        | `L1`       |
| `l2_global_tokens`                      | `rgb_global_tokens`        | `L2`       |
| `l3_global_tokens`                      | `rgb_global_tokens`        | `L3`       |
| `l4_global_tokens`                      | `rgb_global_tokens`        | `L4`       |
| `full_tokens_l1`                        | `rgb_full_tokens`          | `L1`       |
| `aggregator_pooled_l1`                  | `aggregator_pooled`        | `L1`       |
| `l2_aggregator_pooled`                  | `aggregator_pooled`        | `L2`       |
| `l3_aggregator_pooled`                  | `aggregator_pooled`        | `L3`       |
| `l4_aggregator_pooled`                  | `aggregator_pooled`        | `L4`       |
| `duell_l3_aggpool_p2048`                | `aggregator_pooled`        | `L3`       |
| `l3_aggregator_pooled_b200k`            | `aggregator_pooled_b200k`  | `L3`       |
| `duell_l3_aggpool_lottery`              | `aggregator_pooled_b200k`  | `L3`       |
| `duell2_l3_aggpool_b200k_p2048`         | `aggregator_pooled_b200k`  | `L3`       |
| `duell2_l3_aggpool_b200k_tr128`         | `aggregator_pooled_b200k`  | `L3`       |
| `duell2_l3_b200k_tr128_ent3em3`         | `aggregator_pooled_b200k`  | `L3`       |
| `duell2_l3_b200k_tr128_ent3em4`         | `aggregator_pooled_b200k`  | `L3`       |

The values above are the *effective* ones after `extends` resolution: a leaf that
declares neither key inherits both from its parent, which is what makes a thin
probe (`duell_*`, `*_probe`) provably the same arm as the config it extends.

Five arms carry the full L1-L4 ladder (`rgb`, `rgb_pointmap_pose`,
`pointmap_pose`, `rgb_global_tokens`, `aggregator_pooled`); the remaining arms
have an L1 config only. Replay-row size is the reason the token and dense arms
cap `buffer_capacity` in both prod and smoke: a row is 5.6 MB
(`rgb_full_tokens`), 2.8 MB (`rgb_global_tokens`) or 1.6 MB (`pointmap_dense`),
and the buffer preallocates `capacity` rows. The `_base` smoke block only
overrides prod args, so a cap that is missing from the smoke block is not
inherited from anywhere.

`_house_points_pose_l1_live` carries the leading underscore of a shared base
rather than of a variant. Its own arm (the map-as-replacement
`habitat-l1-vggt-house-points-pose` run) was dropped in the adapter-routing
migration, but the two live arms above still `extends:` it for their shared
sbatch and smoke shape. It declares no `adapter:` of its own, so submitting it
would train `house_context_l1`'s arm under the wrong job name; the underscore is
what keeps it out of the variant list.

The legacy `*.sbatch` files this launcher replaced were archived in slice s5
(3D-34) and have since been deleted; recover them from git history if needed.
The remaining hand-written `scripts/r2dreamer/slurm/*.sbatch` files
(`*_actfix`, `*_rerun`, `train_habitat_baseline`, the GNN 50k comparisons) are
historical records of the jobs they ran, not a supported path: they still call
the retired `run.py` dispatcher. New work goes through this launcher.

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

- run with `set -euo pipefail` and `uv run --no-sync python`
- export `WANDB_MODE=offline`, and nothing else: a smoke only answers "does the
  prod configuration start and survive", which it can only do in the prod
  environment. The `PYTHONFAULTHANDLER`/XLA-allocator exports smokes used to add
  made `habitat_sim` SIGABRT in prefill on runs that succeeded under prod (jobs
  6056684, 6056813 against 6056750), so both a green and a red smoke said
  nothing about prod
- when `smoke.assert_file` is set, assert that file exists in the run dir (and
  has ≥ `smoke.assert_min_rows` rows), exiting non-zero otherwise, then print
  `=== Smoke PASS ===`

> **Note on partition choice.** All habitat smokes use `gpu_h100_short`, not
> `dev_gpu_h100`. The `dev_gpu_h100` partition is a single node (`uc3n082`) where
> habitat_sim's OpenGL renderer aborts during prefill, see
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
script: -m src.main       # entrypoint, spliced between the interpreter and the
                          # flags; a repo-relative `.py` path works too.
                          # Always inherited from _base in practice.
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

args:                     # free mapping -> command-line flags; see "How rendering works"
  env: habitat            # inherited from _base
  adapter: rgb            # one key of src.adapters.ADAPTERS
  curriculum: L4          # L1..L4; the leaf's own rung of the ladder
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
  so `gnn_house_points_pose_l1_live → _house_points_pose_l1_live →
  house_context_l1 → _base` collapses into one validated config with no leftover
  `extends` key. Cycles are rejected.
- **`args`** is a free mapping. Each entry renders as a `--flag value` line in
  YAML order, the key verbatim; values containing whitespace, `$`, or `,` are
  quoted (matching the hand-written sbatch convention).
- **Booleans** are the one exception: `full_bf16: true` renders as the bare
  switch `--full_bf16` (no value token), and `full_bf16: false` renders no line
  at all. A boolean-*looking string* is not a boolean and would render as the
  pair `--full_bf16 false`, whose meaning then depends on how the target parser
  declares the flag. Both the quoted `"true"`/`"false"` and the YAML 1.1 words
  `yes`/`no`/`on`/`off` (which the loader hands back as plain strings) are
  rejected at validation time naming the offending key, in `args` and in
  `smoke.args` alike; write the canonical YAML boolean unquoted.
- **`${TIMESTAMP}`** in any rendered value emits a `TIMESTAMP=$(date ...)` line.
- **`${SLURM_JOB_ID}`** and other env vars are expanded by Slurm/bash at runtime.

## Config layout

```
scripts/slurm/
├── configs/
│   ├── _base.yaml                  # shared entrypoint + sbatch + smoke defaults
│   ├── l1_cnn_cap1m.yaml           # extends _base
│   ├── house_context_l1.yaml       # extends _base
│   ├── _house_points_pose_l1_live.yaml   # extends house_context_l1; abstract (`_` prefix)
│   ├── gnn_house_points_pose_l1_live.yaml     # extends _house_points_pose_l1_live
│   └── hybrid_house_points_pose_l1_live.yaml  # extends _house_points_pose_l1_live
├── hooks/
│   └── link_external.sh       # symlink external VGGT repos into a fresh worktree
├── launch.py                  # render + validate
├── launch.sh                  # CLI wrapper around sbatch (multi-variant, --env)
└── train.sbatch               # generic fallback sbatch (manual submit)
```

## Adding a new variant

A new *observation* variant is an adapter plus one `ADAPTERS` row. A new config
here is only a new *job shape* (resources, step budget, smoke gate) over an
existing adapter.

1. Copy an existing config: `cp configs/l1_cnn_cap1m.yaml configs/<name>.yaml`.
2. Edit `job_name`, `output_dir`, `adapter`/`curriculum`, and any `args:` /
   `smoke:` overrides. Reuse a sibling via `extends:` to keep the delta tiny.
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
JAX_PLATFORMS=cpu uv run --no-sync pytest tests/slurm/test_launch.py -v
```

Covers: `--smoke-then-prod` issues the prod submit with
`--dependency=afterok:<smoke_jid>`; fail-fast validation before any sbatch call;
recursive `extends` (incl. cycle rejection); multi-variant sweep submits every
variant; strict-bash and timestamp behavior; and `--env` / `--time` override
handling.

The load-bearing one is the drift guard: every config, in both modes, is
rendered and its argv pushed through `src.launch.parser.build_parser()`. A knob
renamed or deleted in the parser, a value of the wrong type, or an out-of-range
choice fails here instead of on the cluster after the node was allocated.

## Monitoring a submitted job

```bash
squeue -j <jid>                                            # status
tail -f output/<variant>/smoke/slurm-<jid>.out             # smoke logs
tail -f output/<variant>/slurm-<jid>.out                   # prod logs
sacct -j <jid> --format=JobID,State,Elapsed,MaxRSS,ExitCode
```
