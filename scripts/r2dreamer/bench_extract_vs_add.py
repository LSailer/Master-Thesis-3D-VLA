"""GPU microbenchmark: VGGT extract vs HouseContextPoseBuffer.add per step.

Times the two sequential per-step costs of the house-points-pose pipeline on
the real GPU extractor, using the same extractor/adapter config as
``habitat-l1-vggt-house-points-pose``:

    out = extractor.extract(obs)   # (1) VGGT forward + DPT point head  [GPU]
    buffer.add(out, obs)           # (2) host voxel dedup loop          [CPU]

For every step it records:
  * extract_ms      - VGGT forward (content-independent fixed-shape graph)
  * add_full_ms     - buffer.add on the raw 518x518 map (the landmine path)
  * add_stride8_ms  - buffer.add on the stride-8 subsampled map (production)

Two buffers accumulate in parallel over the run so both add paths pay realistic
growing-concatenate cost. ``unique_voxels`` (the host-loop length) is captured
per frame for interpretation.

Run on a GPU node:  uv run python scripts/r2dreamer/bench_extract_vs_add.py
"""

from __future__ import annotations

import dataclasses
import statistics
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from src.buffer.house_context_pose_buffer import HouseContextPoseBuffer
from src.environments.observation import ObservationFrame
from src.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor

IMG = 518
VOXEL_SIZE_M = 0.05  # VGGTHousePointsPoseObsAdapter.DEFAULT_VOXEL_SIZE_M
CONFIDENCE = 1.5  # VGGTHousePointsPoseObsAdapter.DEFAULT_CONFIDENCE_SCORE
MAX_INPUT_POINTS = 4096  # VGGTHousePointsPoseObsAdapter.DEFAULT_MAX_INPUT_POINTS
# Match VGGTEncoder production extractor config (encoders/base.py).
VGGT_TOTAL_BUDGET = 200_000
VGGT_STATIC_BUDGETS = tuple([8333] * 24)

N_STEPS = 30
WARMUP_STEPS = 3  # drop first steps (JIT already warmed, but be safe)


class _CountingBuffer(HouseContextPoseBuffer):
    """Buffer that records the host-loop length (unique voxels) per add."""

    last_unique: int = 0

    def _new_voxel_positions(self, unique_voxels: jax.Array) -> list[int]:
        self.last_unique = int(unique_voxels.shape[0])
        return super()._new_voxel_positions(unique_voxels)


def input_stride(height: int, width: int, budget: int) -> int:
    """Mirror hybrid_adapter._input_stride."""
    total = height * width
    if budget <= 0 or total <= budget:
        return 1
    return max(1, int((total / budget) ** 0.5))


def make_frame(step: int, rng: np.random.Generator) -> np.ndarray:
    """Structured (3, 518, 518) uint8 frame; varies per step. Content does not
    affect extract time (fixed-shape graph) but keeps VGGT geometry non-degenerate.
    """
    axis = np.linspace(0.0, 1.0, IMG, dtype=np.float32)
    gx, gy = np.meshgrid(axis, axis)
    base = (0.5 + 0.5 * np.sin(6.0 * gx + step * 0.1)) * (
        0.5 + 0.5 * np.cos(6.0 * gy + step * 0.13)
    )
    stacked = np.stack([base, np.roll(base, 17, 0), np.roll(base, 31, 1)], axis=0)
    img = (stacked * 255.0).astype(np.int16)
    img = (img + rng.integers(-10, 10, img.shape, dtype=np.int16)).clip(0, 255)
    return np.ascontiguousarray(img.astype(np.uint8))


def strided_inputs(out, img: np.ndarray, stride: int):
    """Stride the (H,W,3) map, (H,W) confidence and (3,H,W) image together."""
    if stride == 1:
        return SimpleNamespace(world_points=out.world_points, confidence=out.confidence), img
    wp = out.world_points[::stride, ::stride, :]
    cf = jnp.asarray(out.confidence)[::stride, ::stride]
    im = np.ascontiguousarray(img[:, ::stride, ::stride])
    return SimpleNamespace(world_points=wp, confidence=cf), im


def main() -> None:
    print(f"jax backend: {jax.default_backend()}  devices: {jax.devices()}")
    stride = input_stride(IMG, IMG, MAX_INPUT_POINTS)
    print(
        f"config: voxel={VOXEL_SIZE_M} m  conf>={CONFIDENCE}  "
        f"max_input_points={MAX_INPUT_POINTS}  stride={stride}  "
        f"budget={VGGT_TOTAL_BUDGET}\n"
    )

    t_load = time.perf_counter()
    extractor = JAXVGGTFeatureExtractor(
        total_budget=VGGT_TOTAL_BUDGET,
        budgets_static=VGGT_STATIC_BUDGETS,
        compute_heads=True,
    )
    print(f"extractor built + warmed in {time.perf_counter() - t_load:.1f} s\n")

    rng = np.random.default_rng(0)
    buf_full = _CountingBuffer(CONFIDENCE, "full", voxel_size_m=VOXEL_SIZE_M)
    buf_str = _CountingBuffer(CONFIDENCE, "stride", voxel_size_m=VOXEL_SIZE_M)

    ext_ms: list[float] = []
    addf_ms: list[float] = []
    adds_ms: list[float] = []

    print(f"{'step':>4} {'extract':>9} {'add_full':>9} {'add_str8':>9} "
          f"{'uniqF':>7} {'uniqS':>6} {'|Bfull':>8} {'Bstr':>6}")
    for step in range(N_STEPS):
        img = make_frame(step, rng)
        obs = ObservationFrame(image=img, is_first=(step == 0), scene_id="bench")

        t0 = time.perf_counter()
        out = extractor.extract(obs)
        out.world_points.block_until_ready()
        out.confidence.block_until_ready()
        out.camera_pose.block_until_ready()
        t1 = time.perf_counter()

        # (2a) raw full-map add
        full_out = SimpleNamespace(
            world_points=out.world_points, confidence=out.confidence
        )
        ta = time.perf_counter()
        pf = buf_full.add(full_out, obs)
        pf.block_until_ready()
        tb = time.perf_counter()

        # (2b) production stride-8 add
        s_out, s_img = strided_inputs(out, img, stride)
        s_obs = dataclasses.replace(obs, image=s_img)
        tc = time.perf_counter()
        ps = buf_str.add(s_out, s_obs)
        ps.block_until_ready()
        td = time.perf_counter()

        e, af, as_ = (t1 - t0) * 1e3, (tb - ta) * 1e3, (td - tc) * 1e3
        nf = buf_full.points_xyz.shape[0] if buf_full.points_xyz is not None else 0
        ns = buf_str.points_xyz.shape[0] if buf_str.points_xyz is not None else 0
        print(f"{step:4d} {e:8.1f}m {af:8.1f}m {as_:8.2f}m "
              f"{buf_full.last_unique:7d} {buf_str.last_unique:6d} {nf:8d} {ns:6d}")
        if step >= WARMUP_STEPS:
            ext_ms.append(e)
            addf_ms.append(af)
            adds_ms.append(as_)

    def stats(xs: list[float]) -> str:
        return f"median {statistics.median(xs):8.1f} ms  (min {min(xs):.1f} / max {max(xs):.1f})"

    print("\n=== per-step medians (excluding first "
          f"{WARMUP_STEPS} warmup steps) ===")
    print(f"  VGGT extract       : {stats(ext_ms)}")
    print(f"  add  full 518^2    : {stats(addf_ms)}")
    print(f"  add  stride-8      : {stats(adds_ms)}")

    me, mf, ms = (statistics.median(ext_ms), statistics.median(addf_ms),
                  statistics.median(adds_ms))
    print("\n=== per-step totals (extract + add) ===")
    print(f"  raw path   : {me + mf:8.1f} ms  (add = {mf / (me + mf) * 100:.0f}% of step)")
    print(f"  stride path: {me + ms:8.1f} ms  (add = {ms / (me + ms) * 100:.0f}% of step)")
    print(f"  add speedup: {mf / ms:.1f}x   step speedup: {(me + mf) / (me + ms):.2f}x")

    steps = 2000
    print(f"\n=== projection ({steps} steps ===")
    print(f"  raw path   : {(me + mf) / 1e3 * steps / 60:6.1f} min")
    print(f"  stride path: {(me + ms) / 1e3 * steps / 60:6.1f} min")


if __name__ == "__main__":
    main()
