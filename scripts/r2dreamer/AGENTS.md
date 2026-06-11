# AGENTS.md — `scripts/r2dreamer/`

Module contract for the R2Dreamer experiment drivers. Scopes the repo-root
[`AGENTS.md`](../../AGENTS.md) to this folder. The *library* these scripts drive is
[`src/r2dreamer/`](../../src/r2dreamer/AGENTS.md).

## Purpose

Runnable drivers and SLURM wrappers for R2Dreamer experiments: curriculum
live-training launchers, profiling, and publication plots. Scripts are thin —
heavy logic lives in `src/r2dreamer/`.

## File map


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
| `profile_training.py` | Per-phase timing of the full loop (env / vggt_forward / vggt_wrapper / jax_upload / wm_inference / buffer_add / wm_training), CNN and VGGT back-to-back. |
| `profile_encoders_3d5253.py` | Isolated encoder cost comparison: VGGT `extract()` vs encoder `forward()` vs `train_step()`, wp_cp vs wp_dense. |
| `plot_curriculum_scaling.py`, `plot_baseline_analysis.py`, `plot_l1_baseline.py` | Publication plots from run `metrics.csv`. |

### `slurm/`
- `train_curriculum_l{1,2,3}{,_actfix,_rerun}.sbatch`, `train_curriculum_l1_vggt_resume*.sbatch`,
  `train_habitat_baseline.sbatch`, `smoke_curriculum_vggt.sbatch`.
- Partitions in use: `dev_gpu_h100` / `gpu_h100_short` (dev & smoke, ~30 min cap),
  `gpu_h100_il` and multi-partition `gpu_h100_il,gpu_h100` (prod, starts on whichever frees first).

## What they read / write

- **Read:** CNN checkpoints (`output/runs/.../step_*.pkl`), curricula
  (`data/curriculum/*.json`), run `metrics.csv`.
- **Write:** live runs → `output/runs/r2dreamer-curriculum-l{N}/…`,
  figures → `output/methods/comparisons/figures/`. Each run dir holds `step_*.pkl`
  and `metrics.csv`.

## Conventions

- Dash-style argparse flags; defaults baked in or passed via sbatch `--export`.
- W&B: live runs → project `…-objectnav`; tags encode issue id, encoder, seed.
  `--no-wandb` for local smoke.
- Run dirs are named by SLURM `JOB_ID`.

## Gotchas / read-this-first

- **Feature dims are load-bearing:** WP/CP `4116`, aggregator `3072`, hybrid `16404`,
  agent obs `64×64`. A mismatch is a shape assertion at train time, not a warning.
- **GPU only via `srun`/sbatch** — never run a training/profiling script directly
  on a login node (see root `AGENTS.md`). In a fresh worktree run `./scripts/setup_worktree.sh` first.
