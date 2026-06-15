#!/usr/bin/env python3
"""Single R2Dreamer run dispatcher — replaces the per-run ``run_jax_*.py`` shims.

Usage::

    uv run python scripts/r2dreamer/run.py <run-id> [train flags...]

The leading positional selects an entry from ``_run_configs.RUN_CONFIGS`` (the
single source of truth for env/Observation Preparation/curriculum/output_dir/
wandb_*). Every flag after it is forwarded verbatim to ``src.main.train`` and
overrides the config's defaults, exactly as the old shims did. Slurm configs
render this positional from their ``run_id:`` field (see
``scripts/slurm/launch.py``).
"""
import sys

import _run_configs


def main(argv: list[str] | None = None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        raise SystemExit(
            "usage: run.py <run-id> [train flags...]\n"
            f"available run ids: {sorted(_run_configs.RUN_CONFIGS)}"
        )
    run_id, train_argv = argv[0], argv[1:]
    if run_id not in _run_configs.RUN_CONFIGS:
        raise SystemExit(
            f"unknown run id: {run_id}\n"
            f"available run ids: {sorted(_run_configs.RUN_CONFIGS)}"
        )
    return _run_configs.launch_run(run_id, argv=train_argv)


if __name__ == "__main__":
    main()
