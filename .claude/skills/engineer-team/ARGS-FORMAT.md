# Args File Format

Per-experiment overrides for the auto-pipeline. Lives at `scripts/pipeline/<name>.args` and is sourced by `launch.sh`. Generic SLURM directives stay in the reusable `scripts/slurm/train.sbatch` — `.args` only overrides what differs per experiment.

## Location

`scripts/pipeline/<name>.args` — one file per experiment. The `<name>` matches the branch (`pipeline/<name>`) and the recap slug.

## Required variables

```bash
# scripts/pipeline/<name>.args
EXPERIMENT_NAME="<name>"                                  # short slug, matches recap + branch
RECAP_PATH="docs/wiki/recaps/<YYYY-MM-DD>-<topic>.md"     # source of criteria
TRAIN_PARTITION="gpu_h100"                                # SLURM partition for train job
TRAIN_TIME="24:00:00"                                     # walltime for train job
TRAIN_CMD="uv run python modules/r2dreamer/launch/train.py --config <name>"
METRICS_PATH="output/runs/<name>/metrics.csv"             # where the run writes its metrics
```

Every variable above is required. `launch.sh` exits non-zero if any are missing.

## Optional variables

Add freely as the experiment needs. These are sourced into the environment of `train.sbatch` via `--export=ALL,...`:

```bash
SEED="42"
WANDB_RUN_GROUP="curriculum-l2-3d"
NUM_STEPS="2_000_000"
```

Keep the file small — anything that's the same across all experiments belongs in `train.sbatch`, not here.

## Rules

- **Shell-sourceable.** No YAML, no JSON. `launch.sh` runs `source <name>.args`, so syntax must be valid bash.
- **Quote everything that could contain spaces or shell metacharacters** (commands, paths). Use double-quotes by default.
- **Do not export.** `launch.sh` re-exports the variables to sbatch via `--export=ALL,...`; double-export is harmless but noisy.
- **No commands.** The file is config, not a script. `TRAIN_CMD` holds the command as a string; running it is the sbatch's job.
- **One per experiment.** If two experiments share 95% of args, that's still two files — copy and diff. Splitting a `.args.common` is premature unless you have ≥3 experiments doing it.

## Example

```bash
# scripts/pipeline/cnn-l2-baseline.args
EXPERIMENT_NAME="cnn-l2-baseline"
RECAP_PATH="docs/wiki/recaps/2026-04-26-cnn-l2-baseline.md"
TRAIN_PARTITION="gpu_h100"
TRAIN_TIME="36:00:00"
TRAIN_CMD="uv run python modules/r2dreamer/launch/train.py --encoder cnn --level 2"
METRICS_PATH="output/runs/cnn-l2-baseline/metrics.csv"
SEED="0"
WANDB_RUN_GROUP="cnn-baselines"
```
