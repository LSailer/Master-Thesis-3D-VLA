"""Random-agent GPU benchmark: VGGT extract vs HouseContextPoseBuffer.add.

Unlike ``bench_extract_vs_add.py`` (synthetic sinusoidal frames), this drives the
real Habitat L1 ObjectNav environment with a RANDOM agent — the same policy the
trainer uses during prefill (``trainer._prefill``):

    action = randint(0, num_actions)
    obs    = env.step(action)

For every step it times the two sequential per-step costs of the
``vggt_house_points_pose`` pipeline (``VGGTHousePointsPoseObsAdapter.transform``):

    out = extractor.extract(obs)   # (1) VGGT forward + DPT point head   [GPU]
    buffer.add(out, obs)           # (2) host voxel dedup + growing cat  [CPU]

Why a random agent matters: ``extract`` is a fixed-shape JIT graph, so its time
is content-independent. ``buffer.add`` is NOT — its host voxel-dedup loop and
growing concatenate scale with how many *distinct* voxels the real scene geometry
yields per frame, so real observations give a truer buffer cost than synthetic
frames.

The production adapter is reused as-is so stride / voxel size / confidence /
max_input_points all match ``habitat-l1-vggt-house-points-pose``.

Run on a GPU node (see the sibling .sbatch, or srun directly):
    srun ... uv run python scripts/r2dreamer/bench_random_agent_extract_add.py
"""

from __future__ import annotations

import argparse
import statistics
import time

import jax
import numpy as np

from src.r2dreamer.adapters.hybrid_adapter import (
    VGGTHousePointsPoseObsAdapter,
    _camera_pose_from_output,
)
from src.r2dreamer.launch.habitat_setup import make_habitat_env
from src.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor

# Match the production extractor config for vggt_house_points_pose
# (src/r2dreamer/encoders/base.py::VGGTEncoder).
VGGT_TOTAL_BUDGET = 200_000
VGGT_STATIC_BUDGETS = tuple([8333] * 24)
RENDER_RESOLUTION = 518


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=40, help="timed + warmup env steps")
    p.add_argument("--warmup", type=int, default=3, help="leading steps dropped from stats")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--curriculum", type=str, default="L1")
    return p.parse_args()


def stats(xs: list[float]) -> str:
    return (
        f"median {statistics.median(xs):8.1f} ms  "
        f"(mean {statistics.fmean(xs):7.1f} / min {min(xs):.1f} / max {max(xs):.1f})"
    )


def main() -> None:
    args = parse_args()
    print(f"jax backend: {jax.default_backend()}  devices: {jax.devices()}")
    print(
        f"config: curriculum={args.curriculum}  render={RENDER_RESOLUTION}  "
        f"steps={args.steps} (warmup {args.warmup})  seed={args.seed}\n"
    )

    t_env = time.perf_counter()
    env = make_habitat_env(
        curriculum=args.curriculum,
        render_resolution=RENDER_RESOLUTION,
        seed=args.seed,
    )
    print(f"habitat env built in {time.perf_counter() - t_env:.1f} s  "
          f"num_actions={env.num_actions}")

    t_load = time.perf_counter()
    extractor = JAXVGGTFeatureExtractor(
        total_budget=VGGT_TOTAL_BUDGET,
        budgets_static=VGGT_STATIC_BUDGETS,
        compute_heads=True,
    )
    # Production adapter: owns per-scene buffers, stride subsampling, voxel/conf
    # config. We reuse its helpers so timing reflects the real pipeline exactly.
    adapter = VGGTHousePointsPoseObsAdapter(extractor)
    print(f"extractor + adapter built in {time.perf_counter() - t_load:.1f} s")
    print(f"adapter: conf>={adapter._confidence_score}  voxel={adapter._voxel_size_m} m  "
          f"max_input_points={adapter._max_input_points}\n")

    rng = np.random.default_rng(args.seed)
    env.reset()
    extractor.reset()

    ext_ms: list[float] = []
    add_ms: list[float] = []

    print(f"{'step':>4} {'extract':>10} {'add':>9} {'unique':>7} {'points':>8} {'done':>4}")
    for step in range(args.steps):
        action = int(rng.integers(0, env.num_actions))
        obs = env.step(action)

        # (1) VGGT forward + DPT point head — GPU
        t0 = time.perf_counter()
        out = extractor.extract(obs)
        out.world_points.block_until_ready()
        out.confidence.block_until_ready()
        out.camera_pose.block_until_ready()
        t1 = time.perf_counter()

        # (2) HouseContextPoseBuffer.add on the production stride-subsampled map
        _camera_pose_from_output(out)  # mirror transform's per-step host work
        buffer = adapter._get_or_create_buffer(obs.scene_id)
        buffer_out, buffer_obs = adapter._subsampled_buffer_input(out, obs)
        unique_before = len(buffer._voxel_keys)
        t2 = time.perf_counter()
        pts = buffer.add(buffer_out, buffer_obs)
        pts.block_until_ready()
        t3 = time.perf_counter()

        e, a = (t1 - t0) * 1e3, (t3 - t2) * 1e3
        npts = int(buffer.points_xyz.shape[0]) if buffer.points_xyz is not None else 0
        new_voxels = len(buffer._voxel_keys) - unique_before
        print(f"{step:4d} {e:9.1f}m {a:8.2f}m {new_voxels:7d} {npts:8d} "
              f"{str(bool(obs.done)):>4}")

        if step >= args.warmup:
            ext_ms.append(e)
            add_ms.append(a)

        if obs.done:
            env.reset()
            extractor.reset()

    print(f"\n=== per-step medians (excluding first {args.warmup} warmup steps) ===")
    print(f"  VGGT extract              : {stats(ext_ms)}")
    print(f"  HouseContextPoseBuffer.add: {stats(add_ms)}")

    me, ma = statistics.median(ext_ms), statistics.median(add_ms)
    total = me + ma
    print("\n=== per-step total (extract + add) ===")
    print(f"  {total:8.1f} ms  "
          f"(extract {me / total * 100:.0f}% / add {ma / total * 100:.0f}%)")

    for horizon in (600, 1200, 2000):
        print(f"  projection {horizon:5d} steps: {total / 1e3 * horizon / 60:6.1f} min")


if __name__ == "__main__":
    main()
