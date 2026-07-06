"""Timing comparison: live house buffer path vs graph-structure ops.

Answers "what would adopting the graph structure cost per step / per refresh
compared to what the buffer path pays today?" All timings on the same device
over the real saved house cloud.

Per-env-step costs (paid every step in production today):
  - buffer.add of a full 518x518 frame (fixed-shape jitted voxel dedup)
  - house_context_array(262_144) even-stride snapshot

Graph-structure costs (candidate additions; per scene or per refresh, NOT
per step):
  - build_knn_graph k=16 over the stored cloud (jaxkd)
  - local_variation_scores (sparse L*X high-pass)
  - gumbel_topk_sample down to the snapshot budget
  - GCN forward pass (3 layers, hidden 64) over the full graph

Run: uv run python scripts/r2dreamer/bench_graph_vs_buffer.py [--iters 20]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src.buffer.house_context_pose_buffer import HouseContextPoseBuffer
from src.r2dreamer.encoders.constants import HOUSE_CONTEXT_MAX_POINTS
from src.prototyp.graph_house_context.graph_gcn import RgbGcnAutoencoder
from src.prototyp.graph_house_context.graph_ops import gumbel_topk_sample, local_variation_scores
from src.prototyp.graph_house_context.knn_graph import build_knn_graph
from src.prototyp.graph_house_context.ply_io import load_ply_xyzrgb

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLY = (
    REPO_ROOT
    / "output/bench/house_context_50steps/bench_50steps_full_1cm/step_00000_context.ply"
)
FRAME_PIXELS = 518 * 518


def timed(fn, iters: int) -> dict[str, float]:
    """Median/min wall time in ms over ``iters`` calls (first call = compile)."""
    compile_start = time.perf_counter()
    jax.block_until_ready(fn())
    compile_ms = (time.perf_counter() - compile_start) * 1e3
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        samples.append((time.perf_counter() - start) * 1e3)
    return {
        "first_call_ms": compile_ms,
        "median_ms": float(np.median(samples)),
        "min_ms": float(np.min(samples)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", type=Path, default=DEFAULT_PLY)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--cuda", action="store_true", help="jaxkd CUDA kernels")
    parser.add_argument(
        "--max-cloud", type=int, default=0,
        help="0 = full cloud; else stride-subsample (CPU syntax checks)",
    )
    parser.add_argument(
        "--capacity-log2", type=int, default=23,
        help="buffer capacity exponent (production: 23)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "outputs/prototype/graph_house_context/bench_graph_vs_buffer.json",
    )
    args = parser.parse_args()

    xyz, rgb = load_ply_xyzrgb(args.ply)
    if args.max_cloud and xyz.shape[0] > args.max_cloud:
        stride = xyz.shape[0] // args.max_cloud
        xyz = xyz[::stride][: args.max_cloud]
        rgb = rgb[::stride][: args.max_cloud]
    num_points = int(xyz.shape[0])
    print(f"device={jax.devices()[0].platform}, cloud={num_points} points")

    # --- Buffer path (per env step) -------------------------------------
    buffer = HouseContextPoseBuffer(
        confidence_score=1.5,
        scene_id="bench",
        capacity=1 << args.capacity_log2,
        hash_table_size=1 << (args.capacity_log2 + 1),
    )
    xyzrgb01 = jnp.concatenate(
        [xyz, jnp.asarray(rgb, dtype=jnp.float32) / 255.0], axis=-1
    )
    buffer.seed_xyzrgb(xyzrgb01)
    print(f"buffer seeded: {buffer.point_count} voxels")

    # Synthetic full-res frame inside the house bbox: realistic voxel churn
    # (mostly re-observed voxels, some new) without needing VGGT.
    key = jax.random.PRNGKey(0)
    low = jnp.min(xyz, axis=0)
    high = jnp.max(xyz, axis=0)
    frame_xyz = jax.random.uniform(
        key, (FRAME_PIXELS, 3), minval=low, maxval=high, dtype=jnp.float32
    )
    frame_rgb = jax.random.randint(key, (FRAME_PIXELS, 3), 0, 256, dtype=jnp.int32)
    frame_conf = jnp.full((FRAME_PIXELS,), 5.0, dtype=jnp.float32)

    from src.buffer.house_context_pose_buffer import _add_frame_to_state

    frame_rgb_u8 = frame_rgb.astype(jnp.uint8)
    state_holder = {"state": buffer._state}

    def buffer_add():
        # The kernel donates the state, so thread it through like production
        # `add` does. After the first call the frame's voxels are all stored,
        # so subsequent calls measure the steady-state pure-dedup path.
        state_holder["state"] = _add_frame_to_state(
            state_holder["state"],
            frame_xyz,
            frame_rgb_u8,
            frame_conf,
            buffer._config,
        )
        return state_holder["state"].size

    results = {"device": jax.devices()[0].platform, "cloud_points": num_points}
    results["buffer_add_frame"] = timed(buffer_add, args.iters)
    buffer._state = state_holder["state"]  # re-attach the donated state
    results["buffer_snapshot_262k"] = timed(
        lambda: buffer.house_context_array(HOUSE_CONTEXT_MAX_POINTS), args.iters
    )

    # --- Graph structure (per scene / per refresh) -----------------------
    graph_iters = max(3, args.iters // 4)
    results["graph_build_knn_k16"] = timed(
        lambda: build_knn_graph(xyz, k=16, cuda=args.cuda).weights, graph_iters
    )
    graph = build_knn_graph(xyz, k=16, cuda=args.cuda)
    results["graph_variation_scores"] = timed(
        lambda: local_variation_scores(xyz, graph), args.iters
    )
    scores = local_variation_scores(xyz, graph)
    budget = min(HOUSE_CONTEXT_MAX_POINTS, num_points)
    results["graph_gumbel_sample"] = timed(
        lambda: gumbel_topk_sample(jax.random.PRNGKey(1), scores, budget), args.iters
    )

    model = RgbGcnAutoencoder(hidden=64, num_layers=3)
    rgb01 = jnp.asarray(rgb, dtype=jnp.float32) / 255.0
    feats = jnp.concatenate(
        [xyz, rgb01, jnp.zeros((num_points, 1), dtype=jnp.float32)], axis=-1
    )
    params = model.init(
        jax.random.PRNGKey(2), feats, graph.senders, graph.receivers,
        graph.weights, num_points,
    )
    @jax.jit
    def gcn_forward_jit(params, feats, senders, receivers, weights):
        # num_points closed over as a Python int -> static under jit
        return model.apply(params, feats, senders, receivers, weights, num_points)

    def gcn_forward():
        return gcn_forward_jit(
            params, feats, graph.senders, graph.receivers, graph.weights
        )

    results["gcn_forward_3x64"] = timed(gcn_forward, args.iters)

    print(f"\n{'operation':<28} {'median':>10} {'min':>10} {'first(comp)':>12}")
    for name, value in results.items():
        if isinstance(value, dict):
            print(
                f"{name:<28} {value['median_ms']:>8.2f}ms {value['min_ms']:>8.2f}ms "
                f"{value['first_call_ms']:>10.1f}ms"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
