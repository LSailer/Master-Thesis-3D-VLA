#!/usr/bin/env python
"""Build the 3D-46 JAX-vs-external(PyTorch) WP/CP comparison table.

Aggregates held-out world-model metrics (mean ± std over the 3 seeds) for the
JAX R2Dreamer (3D-26) and the external PyTorch R2Dreamer (3D-46) into a single
markdown table + a per-seed CSV.

Sources per framework (each a per-seed `heldout_table_row.json`):
  * external: produced by `train_external_offline.py` (this PR), e.g.
      output/3d46-external-offline/wp_cp-seed*/run-*/heldout_table_row.json
  * JAX: produced by `train_offline_ablation.py` (3D-26), e.g.
      output/3d26-offline-ablation/wp_cp-seed*/run-*/heldout_table_row.json
    If those dirs are not on disk, pull the JAX numbers from W&B with
    `--jax-wandb` (reads each run's `final/heldout/*` summary by tag).

Comparable columns (decoder-free on both sides → reconstruction NLL is N/A and
is NOT a head-to-head row):
    dynamics_kl, representation_kl, reward_mse,
    k_step_rollout_k1, k_step_rollout_k5, k_step_rollout_k15

Usage:
    python scripts/r2dreamer/build_offline_comparison.py \\
        --external-glob 'output/3d46-external-offline/wp_cp-seed*/run-*/heldout_table_row.json' \\
        --jax-glob      'output/3d26-offline-ablation/wp_cp-seed*/run-*/heldout_table_row.json' \\
        --out-md   docs/notes/offline-ablation-comparison.md \\
        --out-csv  docs/notes/offline-ablation-comparison.csv

    # JAX numbers from W&B instead of disk:
    python scripts/r2dreamer/build_offline_comparison.py \\
        --external-glob '...heldout_table_row.json' --jax-wandb \\
        --wandb-entity sailer-luca-university-ulm \\
        --wandb-project 3d-vla-objectnav-offline-ablation
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import statistics
from pathlib import Path

# Head-to-head columns (recon NLL excluded — decoder-free on both sides).
COMPARABLE = [
    "dynamics_kl",
    "representation_kl",
    "reward_mse",
    "k_step_rollout_k1",
    "k_step_rollout_k5",
    "k_step_rollout_k15",
]

PRETTY = {
    "dynamics_kl": "Dynamics KL",
    "representation_kl": "Representation KL",
    "reward_mse": "Reward MSE",
    "k_step_rollout_k1": "k-step rollout MSE (k=1)",
    "k_step_rollout_k5": "k-step rollout MSE (k=5)",
    "k_step_rollout_k15": "k-step rollout MSE (k=15)",
}

# W&B summary keys -> table columns (the JAX trainer logs `final/heldout/*`).
WANDB_KEYMAP = {
    "dynamics_kl": "final/heldout/loss/dyn",
    "representation_kl": "final/heldout/loss/rep",
    "reward_mse": "final/heldout/reward_mse",
    "k_step_rollout_k1": "final/heldout/k_step_rollout_mse/k1",
    "k_step_rollout_k5": "final/heldout/k_step_rollout_mse/k5",
    "k_step_rollout_k15": "final/heldout/k_step_rollout_mse/k15",
}

_SEED_RE = re.compile(r"seed[-_]?(\d+)")


def _seed_from_path(path: str) -> str:
    m = _SEED_RE.search(path)
    return m.group(1) if m else path


def load_rows_from_glob(pattern: str) -> dict[str, dict]:
    """Map seed -> table-row dict from heldout_table_row.json files."""
    rows: dict[str, dict] = {}
    for p in sorted(glob.glob(pattern)):
        seed = _seed_from_path(p)
        rows[seed] = json.loads(Path(p).read_text())
    return rows


def load_jax_from_wandb(entity: str, project: str, tags: list[str]) -> dict[str, dict]:
    """Pull JAX per-seed rows from W&B run summaries (`final/heldout/*`)."""
    import wandb

    api = wandb.Api()
    filt = {"tags": {"$all": tags}} if tags else {}
    rows: dict[str, dict] = {}
    for run in api.runs(f"{entity}/{project}", filters=filt):
        summary = run.summary
        row = {col: _to_float(summary.get(key)) for col, key in WANDB_KEYMAP.items()}
        if all(v is None for v in row.values()):
            continue  # not a finished offline run
        seed = _seed_from_path(run.name or run.id)
        rows[seed] = row
    return rows


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def aggregate(rows_by_seed: dict[str, dict]) -> dict[str, dict]:
    """seed->row  ->  metric -> {mean, std, n, per_seed:{seed:val}}."""
    out: dict[str, dict] = {}
    for col in COMPARABLE:
        per_seed = {
            seed: float(row[col])
            for seed, row in rows_by_seed.items()
            if col in row and row[col] is not None and _is_finite(row[col])
        }
        vals = list(per_seed.values())
        out[col] = {
            "mean": statistics.mean(vals) if vals else float("nan"),
            "std": statistics.stdev(vals) if len(vals) >= 2 else 0.0,
            "n": len(vals),
            "per_seed": per_seed,
        }
    return out


def _is_finite(x) -> bool:
    try:
        x = float(x)
        return x == x and x not in (float("inf"), float("-inf"))
    except (TypeError, ValueError):
        return False


def render_markdown(jax_agg: dict, ext_agg: dict, *, meta: dict) -> str:
    lines = [
        "# Offline ablation — JAX vs external (PyTorch) R2Dreamer (WP/CP)",
        "",
        "Held-out world-model metrics on the last 10% of episodes "
        "(mean ± std over seeds {0,1,2}). Lower is better for every column.",
        "",
        "Both frameworks run the decoder-free R2-Dreamer objective "
        "(`rep_loss=\"r2dreamer\"`), so **reconstruction NLL is N/A on both** "
        "and is omitted from the head-to-head.",
        "",
        "| Metric | JAX (3D-26) | external PyTorch (3D-46) |",
        "| -- | -- | -- |",
    ]
    for col in COMPARABLE:
        j, e = jax_agg.get(col, {}), ext_agg.get(col, {})
        lines.append(f"| {PRETTY[col]} | {_cell(j)} | {_cell(e)} |")
    lines += [
        "",
        f"- JAX seeds found: {meta.get('jax_seeds', [])}",
        f"- external seeds found: {meta.get('ext_seeds', [])}",
        f"- JAX source: {meta.get('jax_source', '?')}",
    ]
    if meta.get("wandb_runs"):
        lines.append(f"- W&B runs: {', '.join(meta['wandb_runs'])}")
    lines.append("")
    return "\n".join(lines)


def _cell(agg: dict) -> str:
    if not agg or agg.get("n", 0) == 0:
        return "_pending_"
    return f"{agg['mean']:.4f} ± {agg['std']:.4f} (n={agg['n']})"


def write_csv(path: Path, jax_rows: dict, ext_rows: dict) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["framework", "seed", "metric", "value"])
        for fw, rows in (("jax", jax_rows), ("pytorch_external", ext_rows)):
            for seed, row in sorted(rows.items()):
                for col in COMPARABLE:
                    if col in row and row[col] is not None:
                        w.writerow([fw, seed, col, row[col]])


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--external-glob", required=True,
        help="Glob to external heldout_table_row.json files (one per seed).",
    )
    p.add_argument("--jax-glob", help="Glob to JAX heldout_table_row.json files.")
    p.add_argument("--jax-wandb", action="store_true", help="Fetch JAX rows from W&B.")
    p.add_argument("--wandb-entity", default="sailer-luca-university-ulm")
    p.add_argument("--wandb-project", default="3d-vla-objectnav-offline-ablation")
    p.add_argument("--wandb-tags", default="3d-26,wp_cp")
    p.add_argument("--out-md", default="docs/notes/offline-ablation-comparison.md")
    p.add_argument("--out-csv", default="docs/notes/offline-ablation-comparison.csv")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    ext_rows = load_rows_from_glob(args.external_glob)

    jax_rows: dict[str, dict] = {}
    jax_source = "none"
    if args.jax_glob:
        jax_rows = load_rows_from_glob(args.jax_glob)
        jax_source = f"glob:{args.jax_glob}"
    if not jax_rows and args.jax_wandb:
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        jax_rows = load_jax_from_wandb(args.wandb_entity, args.wandb_project, tags)
        jax_source = f"wandb:{args.wandb_entity}/{args.wandb_project} tags={tags}"

    jax_agg, ext_agg = aggregate(jax_rows), aggregate(ext_rows)
    meta = {
        "jax_seeds": sorted(jax_rows),
        "ext_seeds": sorted(ext_rows),
        "jax_source": jax_source,
    }
    md = render_markdown(jax_agg, ext_agg, meta=meta)

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md)
    write_csv(Path(args.out_csv), jax_rows, ext_rows)

    print(md)
    print(f"\nWrote {out_md} and {args.out_csv}")
    if not ext_rows:
        print("WARNING: no external rows found — runs not finished yet?")
    if not jax_rows:
        print("WARNING: no JAX rows found — pass --jax-glob or --jax-wandb once available.")


if __name__ == "__main__":
    main()
