# AGENTS.md - `scripts/r2dreamer/`

R2Dreamer experiment-driver rules. Inherits repo-root `AGENTS.md`; library code lives in
`src/r2dreamer/`.

## Purpose

Analysis and plot drivers for curriculum experiments. Keep scripts thin; heavy
logic belongs in `src/r2dreamer/`. Launching lives elsewhere: training and
evaluation both run through `python -m src.main` (one parser, `--mode
train|eval`), rendered onto SLURM by `scripts/slurm/launch.sh` from the YAML
configs in `scripts/slurm/configs/`.

## Contracts

- There is no run-id dispatcher and no eval shim anymore: a run is fully
  described by its YAML config (`script: -m src.main` plus `args:` naming
  `adapter`, `curriculum`, and the knobs).
- `adapter` is a key of `src.adapters.ADAPTERS`, not an encoder-type string;
  `src.main.make_adapter` rejects unclaimed variant flags at compose time.
- To add a run, add one `scripts/slurm/configs/*.yaml`. Do not add a Python
  shim.
- Underscore-style argparse flags (`--log_every`); the SLURM launcher renders
  YAML `args:` in underscore style, booleans as bare flags.

## Gotchas

- Shapes come from the adapter's routed fields, not from a shape table: the
  agent is built from one live adapter call on the first frame. Agent obs is
  `64x64` for every variant's replayed image branch.
- Training, profiling, Habitat, VGGT, and smoke runs must use `srun`/sbatch; never
  run them directly on a login node.
- In fresh worktrees, run `./scripts/setup_worktree.sh` before training/eval.
- Live runs write under `output/runs/...`; plot scripts write under
  `output/methods/comparisons/figures/`.

## Useful commands

```bash
uv run python -m src.main --help
./scripts/r2dreamer/run_decoder_probe_overfit_gpu.sh -v
```
