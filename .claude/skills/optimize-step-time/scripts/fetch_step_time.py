"""Fetch step-time statistics for a wandb run.

Prints post-warmup median/p10/p90 of the step-time metric, selected config
fields, and any timing-related keys found in the run summary. Host-side I/O
only — plain NumPy, no JAX, safe on the login node.

Usage:
    .venv/bin/python fetch_step_time.py <entity>/<project>/<run_id> \
        [--key perf/ms_per_step_interval] [--warmup-frac 0.1]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

STEP_KEY_DEFAULT = "perf/ms_per_step_interval"
TIMING_HINTS = ("time", "ms", "fps", "perf", "duration")


def fetch_series(run, key: str, samples: int) -> np.ndarray:
    """Fetches a sampled history series for one metric.

    Uses wandb's server-side sampling (one request) instead of
    ``scan_history``, which pages through the full history and takes
    minutes on multi-million-step runs.

    Args:
      run: A ``wandb.apis.public.Run`` object.
      key: History key to extract, e.g. ``perf/ms_per_step_interval``.
      samples: Number of points to sample across the run, spread evenly
        by the server over the logged steps.

    Returns:
      float64 array of the metric's sampled values ordered by logged
      step; empty array if the key never appears.
    """
    rows = run.history(keys=["_step", key], samples=samples, pandas=False)
    values = [row[key] for row in rows if row.get(key) is not None]
    return np.asarray(values, dtype=np.float64)


def print_stats(series: np.ndarray, key: str, warmup_frac: float) -> None:
    """Prints post-warmup summary statistics for a metric series.

    Args:
      series: Metric values ordered by step.
      key: Metric name, used only for labeling the output.
      warmup_frac: Leading fraction of points dropped as warmup (JIT
        compilation, cache fill) before computing statistics.
    """
    n_warm = int(len(series) * warmup_frac)
    body = series[n_warm:]
    if body.size == 0:
        print(f"  {key}: no data after dropping {n_warm} warmup points")
        return
    p10, p50, p90 = np.percentile(body, [10, 50, 90])
    print(f"  {key}  (n={body.size}, dropped {n_warm} warmup points)")
    print(f"    median: {p50:.1f}   p10: {p10:.1f}   p90: {p90:.1f}")
    print(f"    mean:   {body.mean():.1f} ± {body.std():.1f}")


def main() -> int:
    """Entry point: fetches a run and prints its step-time profile.

    Returns:
      Process exit code: 0 on success, 1 if the run or metric is missing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_path", help="entity/project/run_id")
    parser.add_argument("--key", default=STEP_KEY_DEFAULT)
    parser.add_argument("--warmup-frac", type=float, default=0.1)
    parser.add_argument("--samples", type=int, default=2000)
    args = parser.parse_args()

    import wandb

    api = wandb.Api()
    try:
        run = api.run(args.run_path)
    except Exception as exc:  # noqa: BLE001 — wandb raises several types here
        print(f"error: cannot fetch run {args.run_path!r}: {exc}", file=sys.stderr)
        return 1

    print(f"run: {'/'.join(run.path)}  state={run.state}  name={run.name}")
    print(f"url: {run.url}")

    interesting_cfg = {
        k: v
        for k, v in (run.config or {}).items()
        if any(s in k.lower() for s in ("encoder", "batch", "step", "budget", "dtype"))
    }
    if interesting_cfg:
        print("config (shape-relevant):")
        for k, v in sorted(interesting_cfg.items()):
            print(f"  {k}: {v}")

    series = fetch_series(run, args.key, args.samples)
    print("history:")
    if series.size:
        print_stats(series, args.key, args.warmup_frac)
    else:
        print(f"  {args.key}: not logged in this run")

    timing_summary = {
        k: v
        for k, v in dict(run.summary).items()
        if not k.startswith("_")
        and any(s in k.lower() for s in TIMING_HINTS)
        and isinstance(v, (int, float))
    }
    if timing_summary:
        print("summary timing keys:")
        for k, v in sorted(timing_summary.items()):
            print(f"  {k}: {v:.3f}")

    return 0 if series.size else 1


if __name__ == "__main__":
    sys.exit(main())
