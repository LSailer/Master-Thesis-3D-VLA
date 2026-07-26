# AGENTS.md — `scripts/r2dreamer/`

R2Dreamer experiment-driver rules. Inherits repo-root `AGENTS.md`; library code lives in
`src/r2dreamer/`.

## Purpose

Runnable drivers and SLURM wrappers for curriculum training, evaluation, profiling,
and plots. Keep scripts thin; heavy logic belongs in `src/r2dreamer/`.

## Contracts

- `run.py <run-id> [train flags...]` is the live-training dispatcher.
- `RUN_CONFIGS` in `_run_configs.py` is the single source of truth for run ids and
  `(env, adapter, curriculum, output_dir, wandb_name, wandb_tags)`.
- `adapter` is a key of `src.adapters.ADAPTERS`, not an encoder-type string;
  `launch_run` validates it against that registry before importing `src.main`.
- To add a run, add one `RUN_CONFIGS` entry plus a `scripts/slurm/configs/*.yaml`
  whose `run_id:` names it. Do not add a new Python shim.
- `eval_habitat.py` is the only eval shim; it passes `adapter="rgb"`. For any
  other variant call `src.main.evaluate(adapter=...)` directly.
- Dash-style argparse flags; use `--no-wandb` for local smoke runs.

## Gotchas

- Shapes come from the adapter's routed fields, not from a shape table: the
  agent is built from one live adapter call on the first frame. Agent obs is
  `64×64` for every variant's replayed image branch.
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
