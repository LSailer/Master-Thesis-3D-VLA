"""Experiment 1: build the house-context graph (nodes = xyz, attribute = rgb).

Constructs a symmetrized k-NN graph with Gaussian edge weights over a saved
house point cloud, reports structural/storage statistics, and exports PLYs
for visual inspection in CloudCompare.

Run (CPU login node):
    JAX_PLATFORMS=cpu python -m src.prototyp.graph_house_context.exp1_build_graph
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.prototype_helpers.graph_metrics import table_bytes
from src.prototype_helpers.knn_graph import build_knn_graph, node_degrees
from src.prototype_helpers.ply_io import load_ply_xyzrgb, save_ply_xyzrgb

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLY = (
    REPO_ROOT
    / "output/bench/house_context_50steps/bench_50steps_full_1cm/step_00000_context.ply"
)
DEFAULT_OUT_DIR = REPO_ROOT / "outputs/prototype/graph_house_context/exp1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", type=Path, default=DEFAULT_PLY)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Gaussian bandwidth in meters; default = mean neighbor distance",
    )
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def degree_colors(degrees: jnp.ndarray) -> jnp.ndarray:
    """Map weighted degrees to viridis uint8 RGB for CloudCompare."""
    values = np.asarray(degrees, dtype=np.float32)
    low, high = np.quantile(values, [0.02, 0.98])
    normalized = np.clip((values - low) / max(high - low, 1e-9), 0.0, 1.0)
    colors = plt.get_cmap("viridis")(normalized)[:, :3]
    return jnp.asarray(np.rint(colors * 255.0), dtype=jnp.uint8)


def save_histogram(values: np.ndarray, title: str, xlabel: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.hist(values, bins=80)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("count")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    xyz, rgb = load_ply_xyzrgb(args.ply)
    print(f"loaded {xyz.shape[0]} points from {args.ply}")

    start = time.perf_counter()
    graph = build_knn_graph(xyz, k=args.k, sigma=args.sigma, cuda=args.cuda)
    graph.weights.block_until_ready()
    build_seconds = time.perf_counter() - start
    print(f"built k={args.k} graph in {build_seconds:.1f}s (sigma={graph.sigma:.4f} m)")

    degrees = node_degrees(graph)
    degrees_np = np.asarray(degrees, dtype=np.float32)
    weights_np = np.asarray(graph.weights, dtype=np.float32)

    np.savez(
        args.out_dir / f"graph_k{args.k}.npz",
        senders=np.asarray(graph.senders),
        receivers=np.asarray(graph.receivers),
        weights=weights_np,
        num_nodes=graph.num_nodes,
        k=graph.k,
        sigma=graph.sigma,
    )

    stats = {
        "ply": str(args.ply),
        "build_seconds": build_seconds,
        "k": graph.k,
        "sigma_m": graph.sigma,
        "degree": {
            "min": float(degrees_np.min()),
            "mean": float(degrees_np.mean()),
            "max": float(degrees_np.max()),
        },
        "weight_quantiles": {
            f"p{int(q * 100):02d}": float(np.quantile(weights_np, q))
            for q in (0.05, 0.25, 0.5, 0.75, 0.95)
        },
        "storage": table_bytes(graph.num_nodes, graph.k),
    }
    (args.out_dir / "stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    save_histogram(
        degrees_np,
        "Weighted node degree",
        "degree",
        args.out_dir / "degree_hist.png",
    )
    save_histogram(
        weights_np,
        "Edge weights exp(-d^2/sigma^2)",
        "weight",
        args.out_dir / "weight_hist.png",
    )

    save_ply_xyzrgb(args.out_dir / "nodes_rgb.ply", xyz, rgb)
    save_ply_xyzrgb(args.out_dir / "nodes_degree.ply", xyz, degree_colors(degrees))

    storage = stats["storage"]
    print(
        f"node table {storage['node_table_bytes'] / 1e6:.1f} MB, "
        f"edge table (implicit senders) "
        f"{storage['edge_table_implicit_senders_bytes'] / 1e6:.1f} MB"
    )
    print(f"outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
