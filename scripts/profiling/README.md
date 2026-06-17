# Profiling Scripts

This folder contains manual profiling entrypoints for GPU nodes. Use these when
you need timing evidence that is narrower than a full training run.

Do not run VGGT, Habitat, or JAX GPU profiling directly on the login node. Start
an interactive Slurm allocation with `srun`, then run the Python command inside
that allocation from the repository root.

## Quick Start With Slurm Launcher

Prefer the YAML-backed Slurm launcher for repeatable profiling jobs:

```bash
scripts/slurm/launch.sh profile_encoder_cost --smoke
scripts/slurm/launch.sh profile_training_vggt --smoke
scripts/slurm/launch.sh profile_agg_pipeline --smoke
```

For longer profiling windows, use the default prod mode with a walltime
override:

```bash
scripts/slurm/launch.sh profile_training_vggt --time 01:30:00
```

The launcher writes Slurm logs and profiling artifacts under
`output/profiling/<profile-name>/`.

## Interactive `srun`

Use a short H100 allocation for interactive profiling:

```bash
srun --partition=gpu_h100_short --gres=gpu:1 --ntasks=1 \
  --cpus-per-task=8 --mem=64G --time=00:30:00 --pty bash
```

Inside the allocation:

```bash
cd /home/ul/ul_student/ul_hfj15/Master-Thesis-3D-VLA/worktrees/3d-77-l1-vggt-house-context
./scripts/setup_worktree.sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
bash scripts/slurm/hooks/link_external.sh
```

Prefer the worktree-local interpreter:

```bash
.venv/bin/python -m scripts.profiling.profile_encoders_3d5253
```

Use `uv run --no-sync python` only when the worktree does not have a linked
`.venv/` yet:

```bash
uv run --no-sync python -m scripts.profiling.profile_encoders_3d5253
```

For a one-shot run from the login shell:

```bash
srun --partition=gpu_h100_short --gres=gpu:1 --ntasks=1 \
  --cpus-per-task=8 --mem=64G --time=00:30:00 \
  bash -lc 'cd /home/ul/ul_student/ul_hfj15/Master-Thesis-3D-VLA/worktrees/3d-77-l1-vggt-house-context && \
    ./scripts/setup_worktree.sh && \
    export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 && \
    bash scripts/slurm/hooks/link_external.sh && \
    .venv/bin/python -m scripts.profiling.profile_encoders_3d5253'
```

## Active Profilers

### Encoder Cost

Isolates VGGT extraction, `train_step`, and encoder forward cost for WP/CP vs
dense WP variants.

```bash
.venv/bin/python -m scripts.profiling.profile_encoders_3d5253
```

### Full Training Loop Phases

Profiles the CNN and VGGT training loop phases: env step, VGGT forward/wrapper,
JAX upload, world-model inference, replay add, and world-model training.

```bash
.venv/bin/python -m scripts.profiling.profile_training \
  --encoder vggt \
  --prefill_steps 200 \
  --acting_steps 200 \
  --output_dir output/profiling/profile_training
```

### Aggregator-MLP Pipeline

Profiles the old aggregator-MLP acting/training pipeline. Keep this for
comparisons against 3D-50/3D-52 era runs.

```bash
.venv/bin/python scripts/profiling/profile_pipeline_aggregator_mlp.py \
  --prefill 200 \
  --warmup 20 \
  --measure 100 \
  --out output/profiling/pipeline_aggregator_mlp_${SLURM_JOB_ID:-local}.json
```

### VGGT Streaming

Wrappers around the VGGT JAX streaming profiler and benchmark:

```bash
.venv/bin/python -m scripts.profiling.profile_vggt_streaming
.venv/bin/python -m scripts.profiling.benchmark_vggt_streaming
```

## Diagnostics

`diagnostics/` contains artifact dump and verification scripts used to reproduce
older evidence, mostly 3D-48 world-point pooling figures. They are not required
to start profiling.

Dense-vs-pooled world-point dump:

```bash
.venv/bin/python -m scripts.profiling.diagnostics.dump_wp_dense_vs_pooled \
  --frames tests/r2dreamer/launch/fixtures/sample_habitat_obs.npz \
  --frame-index 0 \
  --out output/3d48/wp_dense_vs_pooled.npz
```

Point-map semantics verification:

```bash
.venv/bin/python -m scripts.profiling.diagnostics.verify_pointmap_is_xyz \
  --npz output/3d48/wp_dense_vs_pooled.npz \
  --out output/3d48/verify_pointmap.png
```

Synthetic world-point dump:

```bash
.venv/bin/python -m scripts.profiling.diagnostics.dump_vggt_points \
  --out output/vggt_points.npz
```

## Full Training Smokes

For normal experiment smokes and production runs, use the Slurm YAML launcher
instead of these profiling scripts:

```bash
scripts/slurm/launch.sh house_context_l1 --smoke
scripts/slurm/launch.sh house_context_l1 --prod
```

For a prod-shaped profiling window, override only the Slurm walltime while
keeping the same run config:

```bash
scripts/slurm/launch.sh house_context_l1 --time 04:00:00
```
