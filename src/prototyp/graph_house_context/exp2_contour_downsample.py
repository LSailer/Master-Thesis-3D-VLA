"""Experiment 2: contour-aware graph downsampling vs even-stride resampling.

Implements the feature-preserving resampling of "Graph Spectral Image
Processing" chapter 7, section 7.4.2.1 (Chen et al. 2018): sample points with
probability proportional to the high-pass graph-filter response
``pi_i ~ ||(L X)_i||``, so geometric contours (walls, edges, corners) survive
aggressive reduction. The baseline is the live pipeline's even-stride
``HouseContextPoseBuffer.resample_xyzrgb``, evaluated at the same point
budgets with one-sided chamfer distances against the full cloud.

Run (CPU login node):
    JAX_PLATFORMS=cpu python -m src.prototyp.graph_house_context.exp2_contour_downsample
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.buffer.house_context_pose_buffer import HouseContextPoseBuffer
from src.prototype_helpers.graph_metrics import chamfer_distances
from src.prototype_helpers.graph_ops import gumbel_topk_sample, local_variation_scores
from src.prototype_helpers.knn_graph import build_knn_graph
from src.prototype_helpers.ply_io import load_ply_xyzrgb, save_ply_xyzrgb

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLY = (
    REPO_ROOT
    / "output/bench/house_context_50steps/bench_50steps_full_1cm/step_00000_context.ply"
)
DEFAULT_OUT_DIR = REPO_ROOT / "outputs/prototype/graph_house_context/exp2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", type=Path, default=DEFAULT_PLY)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument(
        "--budgets",
        type=lambda text: [int(value) for value in text.split(",")],
        default=[4096, 16384, 65536],
    )
    parser.add_argument(
        "--rgb-weight",
        type=float,
        default=0.0,
        help="add rgb_weight * ||(L rgb)_i|| color-edge term to the scores",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def score_colors(scores: jnp.ndarray) -> jnp.ndarray:
    """Map sampling scores to inferno uint8 RGB (log scale for visibility)."""
    values = np.log10(np.asarray(scores, dtype=np.float32) + 1e-12)
    low, high = np.quantile(values, [0.02, 0.98])
    normalized = np.clip((values - low) / max(high - low, 1e-9), 0.0, 1.0)
    colors = plt.get_cmap("inferno")(normalized)[:, :3]
    return jnp.asarray(np.rint(colors * 255.0), dtype=jnp.uint8)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    xyz, rgb = load_ply_xyzrgb(args.ply)
    print(f"loaded {xyz.shape[0]} points from {args.ply}")

    graph = build_knn_graph(xyz, k=args.k, cuda=args.cuda)
    scores = local_variation_scores(
        xyz, graph, rgb=rgb if args.rgb_weight > 0.0 else None,
        rgb_weight=args.rgb_weight,
    )
    save_ply_xyzrgb(args.out_dir / "scores.ply", xyz, score_colors(scores))

    xyzrgb = jnp.concatenate(
        [xyz, jnp.asarray(rgb, dtype=jnp.float32) / 255.0], axis=-1
    )
    rgb_uint8 = jnp.asarray(rgb, dtype=jnp.uint8)

    # Structural-fidelity target: the top-decile local-variation points are
    # the scene's contours (wall edges, corners, object boundaries) — the
    # detail Ch 7 s7.4.2.1 sampling is designed to preserve.
    contour_threshold = jnp.quantile(scores, 0.9)
    contour_region_xyz = xyz[scores >= contour_threshold]

    rows: list[dict[str, float | int | str]] = []
    key = jax.random.PRNGKey(args.seed)
    for budget in args.budgets:
        key, sample_key = jax.random.split(key)
        contour_indices = gumbel_topk_sample(sample_key, scores, m=budget)
        contour_xyz = xyz[contour_indices]
        save_ply_xyzrgb(
            args.out_dir / f"contour_{budget}.ply",
            contour_xyz,
            rgb_uint8[contour_indices],
        )

        stride_rows = HouseContextPoseBuffer.resample_xyzrgb(xyzrgb, budget)
        stride_xyz = stride_rows[:, :3]
        save_ply_xyzrgb(
            args.out_dir / f"stride_{budget}.ply",
            stride_xyz,
            jnp.asarray(
                jnp.rint(jnp.clip(stride_rows[:, 3:], 0.0, 1.0) * 255.0),
                dtype=jnp.uint8,
            ),
        )

        for method, sample_xyz in (
            ("contour", contour_xyz),
            ("stride", stride_xyz),
        ):
            # Chamfer stays on the pure-JAX jaxkd path: the CUDA extension
            # segfaults on k=1 queries against small sample trees (H100,
            # jaxkd 0.1.2), and pure JAX runs on the GPU device anyway.
            _, full_to_sample = chamfer_distances(sample_xyz, xyz)
            contour_to_sample, _ = chamfer_distances(contour_region_xyz, sample_xyz)
            rows.append(
                {
                    "budget": budget,
                    "method": method,
                    "chamfer_full_to_sample_m": full_to_sample,
                    "chamfer_contour_to_sample_m": contour_to_sample,
                }
            )
            print(
                f"budget {budget:>6} {method:>7}: "
                f"coverage full->sample {full_to_sample:.6f} m, "
                f"contour->sample {contour_to_sample:.6f} m"
            )

    (args.out_dir / "metrics.json").write_text(
        json.dumps({"config": vars(args) | {"ply": str(args.ply), "out_dir": str(args.out_dir)}, "rows": rows}, indent=2, default=str),
        encoding="utf-8",
    )
    with (args.out_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_chamfer(rows, args.budgets, args.out_dir / "chamfer_vs_budget.png")
    print(f"outputs in {args.out_dir}")


def plot_chamfer(
    rows: list[dict[str, float | int | str]], budgets: list[int], path: Path
) -> None:
    # Samples are subsets of the full cloud, so the sample->full direction is
    # identically zero; report overall coverage and contour-region fidelity.
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    panels = (
        ("chamfer_full_to_sample_m", "Overall coverage (full -> sample)"),
        ("chamfer_contour_to_sample_m", "Contour fidelity (contours -> sample)"),
    )
    for axis, (metric, title) in zip(axes, panels, strict=True):
        for method in ("contour", "stride"):
            values = [row[metric] for row in rows if row["method"] == method]
            axis.plot(budgets, values, marker="o", label=method)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("point budget")
        axis.set_ylabel("chamfer distance [m]")
        axis.set_title(title)
        axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


if __name__ == "__main__":
    main()
