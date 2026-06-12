"""Per-phase latency breakdown for the JAX VGGT extractor.

Reproduces the body of ``JAXVGGTFeatureExtractor.extract`` with explicit
``block_until_ready()`` between each major component so wall time can be
attributed to the input prep, aggregator, camera head, point head, and the
pool+numpy transfer separately. Use this when ``benchmark_streaming.py``
shows you a per-frame number and you need to know which component to
optimize next.

Run::

    uv run python -m src.vggt.jax.profile_streaming --n-frames 10
    uv run python -m src.vggt.jax.profile_streaming --n-frames 5 --trace /tmp/vggt_trace

The optional ``--trace DIR`` flag writes a Perfetto-compatible JAX profile
trace for one frame; open ``DIR/plugins/profile/...`` in
``chrome://tracing`` or in TensorBoard's profile tab for a kernel-level
view.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src.shared.profiling import make_synthetic_rgb_frame
from src.vggt.jax.feature_extractor import (
    JAXVGGTFeatureExtractor,
    _adaptive_avg_pool_518_to_37,
)


WARMUP_FRAMES = 3
PHASES = ("input_prep", "aggregator", "camera_head", "point_head", "pool_transfer")


def _profile_one_frame(ext: JAXVGGTFeatureExtractor, rgb: np.ndarray) -> dict[str, float]:
    """Run one streaming frame, returning per-phase wall time in ms.

    Mirrors ``JAXVGGTFeatureExtractor.extract`` but inserts
    ``block_until_ready()`` between phases so each timer captures only its
    own segment. Mutates the extractor's cache state exactly as ``extract``
    would (so this is suitable for streaming sequences, not just isolated
    frames).
    """
    t = {}

    # 1. Input prep: numpy uint8 -> JAX bf16 on device.
    t0 = time.perf_counter()
    img = (jnp.asarray(rgb, dtype=jnp.float32) / 255.0).astype(ext._dtype)
    images = img[None, None]
    images = jax.device_put(images, ext._device)
    images.block_until_ready()
    t["input_prep"] = (time.perf_counter() - t0) * 1000.0

    # 2. Aggregator forward (24 alternating frame/global blocks + cache update).
    # Routes through the extractor's jitted wrapper so we measure the real
    # streaming-path cost, not a fresh apply without compilation cache.
    t0 = time.perf_counter()
    if ext._past_kvs_padded is None:
        ext._past_kvs_padded = [
            ext._new_padded_cache_entry() for _ in range(ext._agg_depth)
        ]
    if ext._last_scores is None:
        ext._last_scores = jnp.zeros((ext._agg_depth,), dtype=jnp.float32)
    ls_np = np.asarray(ext._last_scores)
    budgets_static = ext._compute_static_budgets(ls_np)
    out_list, patch_start_idx, ext._past_kvs_padded, ext._last_scores = (
        ext._aggregator_apply(
            ext._agg_params,
            images,
            ext._past_kvs_padded,
            ext._frame_idx == 0,  # is_first_frame bool — 2 compiles total
            ext._total_budget,
            ext._last_scores,
            True,
            budgets_static,
        )
    )
    out_list[-1].block_until_ready()
    if ext._past_kvs_padded is not None:
        ext._past_kvs_padded[-1][0].block_until_ready()
    t["aggregator"] = (time.perf_counter() - t0) * 1000.0

    # 3. Camera head (iterative refiner + its own cache).
    # Routes through the extractor's jitted wrapper with padded 3-tuple
    # cache so we measure the real streaming-path cost.
    t0 = time.perf_counter()
    if ext._past_kvs_camera is None:
        ext._past_kvs_camera = [
            ext._new_padded_camera_entry() for _ in range(ext._cam_depth)
        ]
    pose_list, ext._past_kvs_camera = ext._camera_head_apply(
        ext._cam_params,
        out_list,
        ext._past_kvs_camera,
    )
    pose_enc = pose_list[-1]
    camera_pose = pose_enc[:, 0, :]
    camera_pose.block_until_ready()
    t["camera_head"] = (time.perf_counter() - t0) * 1000.0

    # 4. Point head (DPT, jit-compiled, fixed shapes).
    t0 = time.perf_counter()
    pts3d, _ = ext._point_head_apply(ext._pt_params, out_list, images, int(np.asarray(patch_start_idx)))
    pts3d = pts3d[:, 0]
    pts3d.block_until_ready()
    t["point_head"] = (time.perf_counter() - t0) * 1000.0

    # 5. Pool 518->37 + JAX -> numpy host transfer.
    t0 = time.perf_counter()
    world_points = _adaptive_avg_pool_518_to_37(pts3d)
    world_points_np = np.asarray(world_points[0], dtype=np.float32)
    camera_pose_np = np.asarray(camera_pose[0], dtype=np.float32)
    t["pool_transfer"] = (time.perf_counter() - t0) * 1000.0

    ext._frame_idx += 1

    # Touch outputs so the consumer can verify shapes if needed.
    assert world_points_np.shape == (37, 37, 3)
    assert camera_pose_np.shape == (9,)

    return t


def run(n_frames: int, dtype_name: str, trace_dir: Path | None) -> None:
    dtype = {"bf16": jnp.bfloat16, "fp32": jnp.float32}[dtype_name]
    print(f"Building extractor (dtype={dtype_name}) ...", flush=True)
    ext = JAXVGGTFeatureExtractor(device="cuda", dtype=dtype)

    # Warmup.
    print(f"Warmup ({WARMUP_FRAMES} frames) ...", flush=True)
    ext.reset()
    for i in range(WARMUP_FRAMES):
        _profile_one_frame(ext, make_synthetic_rgb_frame(i))

    # Optional trace capture (one extra frame, after warmup is hot).
    if trace_dir is not None:
        trace_dir.mkdir(parents=True, exist_ok=True)
        print(f"Capturing JAX profile trace -> {trace_dir} ...", flush=True)
        with jax.profiler.trace(str(trace_dir)):
            _profile_one_frame(ext, make_synthetic_rgb_frame(999))

    # Measurement: fresh cache, then n_frames timed frames.
    ext.reset()
    rows: list[dict[str, float]] = []
    for i in range(n_frames):
        rows.append(_profile_one_frame(ext, make_synthetic_rgb_frame(1000 + i)))
        print(f"  frame {i+1}/{n_frames}: total={sum(rows[-1].values()):.1f}ms", flush=True)

    # Aggregate.
    print("\nPer-phase wall time (ms), n={} frames, dtype={}".format(n_frames, dtype_name))
    print("-" * 78)
    print(f"{'phase':<18} {'mean':>10} {'median':>10} {'std':>10} {'min':>10} {'max':>10} {'%':>6}")
    print("-" * 78)
    totals = np.array([sum(r.values()) for r in rows])
    for phase in PHASES:
        vals = np.array([r[phase] for r in rows])
        share = 100.0 * vals.mean() / totals.mean()
        print(
            f"{phase:<18} {vals.mean():>10.2f} {np.median(vals):>10.2f} "
            f"{vals.std():>10.2f} {vals.min():>10.2f} {vals.max():>10.2f} {share:>5.1f}%"
        )
    print("-" * 78)
    print(
        f"{'TOTAL':<18} {totals.mean():>10.2f} {np.median(totals):>10.2f} "
        f"{totals.std():>10.2f} {totals.min():>10.2f} {totals.max():>10.2f} {100.0:>5.1f}%"
    )

    del ext
    gc.collect()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-frames", type=int, default=10, help="Frames to time after warmup.")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument(
        "--trace",
        type=Path,
        default=None,
        help="Optional directory for jax.profiler.trace() output (one extra frame).",
    )
    args = p.parse_args()
    run(args.n_frames, args.dtype, args.trace)


if __name__ == "__main__":
    main()
