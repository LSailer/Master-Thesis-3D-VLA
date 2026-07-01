# AGENTS.md — `scripts/r2dreamer/`

R2Dreamer experiment-driver rules. Inherits repo-root `AGENTS.md`; library code lives in
`src/r2dreamer/`.

## Purpose

Runnable drivers and SLURM wrappers for curriculum training, evaluation, profiling,
and plots. Keep scripts thin; heavy logic belongs in `src/r2dreamer/`.

## Contracts

- `run.py <run-id> [train flags...]` is the live-training dispatcher.
- `RUN_CONFIGS` in `_run_configs.py` is the single source of truth for run ids and
  `(env, encoder, curriculum, output_dir, wandb_name, wandb_tags)`.
- To add a run, add one `RUN_CONFIGS` entry plus a `scripts/slurm/configs/*.yaml`
  whose `run_id:` names it. Do not add a new Python shim.
- Eval shims are `eval_habitat.py` and `eval_habitat_vggt.py`.
- Dash-style argparse flags; use `--no-wandb` for local smoke runs.

## Gotchas

- Feature dims are load-bearing: WP/CP `4116`, aggregator `3072`, hybrid `16404`,
  agent obs `64×64`.
- Training, profiling, Habitat, VGGT, and smoke runs must use `srun`/sbatch; never
  run them directly on a login node.
- In fresh worktrees, run `./scripts/setup_worktree.sh` before training/eval.
- Live runs write under `output/runs/...`; plot scripts write under
  `output/methods/comparisons/figures/`.

## Useful commands

```bash
uv run python scripts/r2dreamer/run.py <run-id> --help
./scripts/r2dreamer/run_decoder_probe_overfit_gpu.sh -v
```
