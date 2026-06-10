# AGENTS.md — `scripts/r2dreamer/`

Module contract for the R2Dreamer experiment drivers. Scopes the repo-root
[`AGENTS.md`](../../AGENTS.md) to this folder. The *library* these scripts drive is
[`src/r2dreamer/`](../../src/r2dreamer/AGENTS.md).

## Purpose

Runnable drivers and SLURM wrappers for R2Dreamer experiments: offline buffer
collection (with VGGT feature extraction), offline-only training (JAX **and** external
PyTorch), curriculum live-training launchers, cross-framework comparison, profiling, and
publication plots. Scripts are thin — heavy logic lives in `src/r2dreamer/`.

## File map

### Data collection
| File | Role |
|------|------|
| `collect_offline_buffer.py` | Roll out a CNN-policy checkpoint, extract VGGT WP/CP + aggregator features, write a crash-tolerant offline buffer (flush at episode boundaries). Flags: `--checkpoint --n-steps --collect-seed --out-dir --split --curriculum-path --vggt-{total,static}-budget --profile`. |
| `make_synthetic_offline_buffer.py` | Tiny random buffer for smoke-testing the offline pipeline. `--out-dir --n-steps --n-episodes --seed`. |

### Offline-only training (no live env)
| File | Role |
|------|------|
| `train_offline_ablation.py` | **3D-26** JAX offline trainer; encoder ∈ {`wp_cp`, `aggregator`} × seed ∈ {0,1,2}. Enforces a shared-hyperparameter fairness contract (batch=16, seq_len=64, imag_horizon=15). Logs held-out WM metrics. |
| `train_external_offline.py` | **3D-45/46** offline trainer for the *external PyTorch* R2Dreamer. Same sampling/logging as the JAX run → apples-to-apples baseline. **Requires `external/r2dreamer/.venv`.** |

### Live training launcher (single dispatcher → `src.r2dreamer.launch.train`)
`run.py <run-id> [train flags...]` is the one entrypoint for every live training
run (the per-run `run_jax_*.py` shims were folded into it). The per-run `(env,
encoder, curriculum, output_dir, wandb_name, wandb_tags)` lives in the
`RUN_CONFIGS` table in [`_run_configs.py`](_run_configs.py) (single source of
truth; `launch_run` validates the encoder against `encoder_registry` at launch).
Run ids: `habitat-l{1,2,3,4}-cnn`, `habitat-l{1,2,3,4}-vggt`,
`habitat-l1-{hybrid,vggt-aggregator-mlp,vggt-wp-cp-64,vggt-wp-dense}`,
`crafter-cnn`. **To add a run:** add one `RUN_CONFIGS` entry + a
`scripts/slurm/configs/*.yaml` whose `run_id:` names it — no new Python file.
Slurm configs render `run.py <id>` from their `run_id` field (the shared
`script: run.py` is inherited from `_base`); legacy `*.sbatch` call `run.py <id>`
directly. Eval shims: `eval_habitat.py`, `eval_habitat_vggt.py`. Validation
shims: `run_parity_training.py`, `run_benchmark.py`.

### Curriculum / analysis / profiling
| File | Role |
|------|------|
| `build_l3_heldout_curriculum.py` | Build a held-out-house L3 chair-only eval curriculum disjoint from L3 train houses. |
| `build_curriculum_slides.py` | Aggregate curriculum-progression metrics into slides/plots. |
| `build_offline_comparison.py` | Aggregate held-out WM metrics (mean±std over 3 seeds) into the 3D-46 JAX-vs-PyTorch table (`--out-md --out-csv`, or read JAX from W&B). |
| `profile_training.py` | Per-phase timing of the full loop (env / vggt_forward / vggt_wrapper / jax_upload / wm_inference / buffer_add / wm_training), CNN and VGGT back-to-back. |
| `profile_encoders_3d5253.py` | Isolated encoder cost comparison: VGGT `extract()` vs encoder `forward()` vs `train_step()`, wp_cp vs wp_dense. |
| `plot_curriculum_scaling.py`, `plot_baseline_analysis.py`, `plot_l1_baseline.py` | Publication plots from run `metrics.csv`. |

### `slurm/`
- `submit_offline_ablation.sh {dev|prod|smoke} {wp_cp|aggregator} {0,1,2} [--dry-run]`
  and `submit_offline_ablation_sweep.sh` (all 6 runs) → `train_offline_ablation.sbatch`.
- `submit_external_offline.sh {dev|prod|smoke} {0,1,2} [encoder] [--dry-run]`
  → `train_external_offline.sbatch` (pins `external/r2dreamer/.venv/bin/python`).
- `train_curriculum_l{1,2,3}{,_actfix,_rerun}.sbatch`, `train_curriculum_l1_vggt_resume*.sbatch`,
  `train_habitat_baseline.sbatch`, `smoke_curriculum_vggt.sbatch`.
- Partitions in use: `dev_gpu_h100` / `gpu_h100_short` (dev & smoke, ~30 min cap),
  `gpu_h100_il` and multi-partition `gpu_h100_il,gpu_h100` (prod, starts on whichever frees first).

## What they read / write

- **Read:** CNN checkpoints (`output/runs/.../step_*.pkl`), offline buffers
  (`data/offline_buffer*/`), curricula (`data/curriculum/*.json`), run `metrics.csv`.
- **Write:** offline runs → `output/3d26-offline-ablation/{encoder}-seed{seed}/run-{JOB_ID}/`,
  external → `output/3d46-external-offline/…`, live → `output/runs/r2dreamer-curriculum-l{N}/…`,
  figures → `output/methods/comparisons/figures/`. Each run dir holds `step_*.pkl`,
  `metrics.csv`, and (offline) `heldout_table_row.json`.
- **Offline buffer layout:** `trajectory_skeleton.npz` (action/reward/done/episode_id),
  `z_wp_cp.npz` (4116-d), `z_aggregator.npz`, `collection_metadata.json`, `rollout_log.jsonl`.

## Conventions

- Dash-style argparse flags; defaults baked in or passed via sbatch `--export`.
- W&B: offline runs → project `…-offline-ablation`, live runs → `…-objectnav`;
  tags encode issue id, encoder, seed. `--no-wandb` for local smoke.
- Run dirs are named by SLURM `JOB_ID`; held-out eval cadence via `--heldout-eval-every`.

## Smoke / profiling contract

Every new or modified smoke/profiling path in this folder must report enough
timing to decide whether the corresponding full experiment is feasible at the
target training budget. For ObjectNav experiments, the default budget estimate is
`2,000,000` environment steps unless the issue explicitly specifies another
target.

At minimum, smoke/profile output must include:

- environment steps/sec or per-step wall time;
- train steps/sec or train-step wall time when training is part of the path;
- feature-extraction time when VGGT or another external encoder runs online;
- replay write/sample overhead when replay is part of the path;
- an estimated wall-clock time for the target step budget;
- an explicit yes/no feasibility statement for the current bwUniCluster
  allocation, including the limiting factor when the answer is no.

If the smoke/profile run cannot produce one of these measurements, it must say
which measurement is missing and why. Do not queue a production run from
`scripts/r2dreamer/` before this timing feedback is available.

## Gotchas / read-this-first

- **External PyTorch runs use a different venv.** `train_external_offline.py` and
  `submit_external_offline.sh` MUST run under `external/r2dreamer/.venv/bin/python` — the
  main `.venv` is JAX-only and will fail on import. The sbatch template hard-codes this.
- **3D-26 fairness contract.** `train_offline_ablation.py` deliberately overrides the
  aggregator encoder's in-code defaults to the shared (batch=16, seq_len=64) values — only
  `--encoder` and `--seed` may differ across the 6 runs.
- **Feature dims are load-bearing:** WP/CP `4116`, aggregator `3072`, hybrid `16404`,
  agent obs `64×64`. A mismatch is a shape assertion at train time, not a warning.
- **Smoke first.** `make_synthetic_offline_buffer.py` → `train_offline_ablation.py … --no-wandb
  --skip-heldout-eval`, or `submit_*.sh dev …`, before queueing prod sweeps.
- **GPU only via `srun`/sbatch** — never run a collection/training/profiling script directly
  on a login node (see root `AGENTS.md`). In a fresh worktree run `./scripts/setup_worktree.sh` first.
- **Buffers are crash-tolerant but truncating:** on crash the buffer is valid up to the last
  complete episode; training clips `z_*` arrays to the skeleton length.
