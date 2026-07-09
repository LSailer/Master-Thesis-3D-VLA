"""Per-stage breakdown of HouseContextPoseBuffer.add on a random agent.

Follow-up to ``bench_random_agent_extract_add.py``, which found ``add`` is ~92%
of the extract+add cost and wildly variable (5-1845 ms). This script instruments
each internal stage of ``add`` (using the real production stride input from a
random agent on Habitat L1) to attribute that cost line-by-line:

    flatten  : reshape/moveaxis of the (H,W,3) world map + RGB + confidence
    mask     : finite & confidence>=thr boolean mask
    count    : bool(jnp.any(mask)) host sync
    gather   : flat_points[mask] / flat_rgb[mask]  (data-dependent output size)
    quantize : floor(points / voxel) -> int32 voxel keys
    unique   : jnp.unique(quantized, axis=0, return_index=True)   <-- suspect
    newpos   : unique_voxels.tolist() sync + Python set-membership loop
    append   : representative gather + growing jnp.concatenate

Each stage is block_until_ready()'d so GPU work is attributed to the right stage.
The stages call the SAME helper methods as the real ``add`` (and mutate the same
buffer state), so accumulation stays identical to production.

Run on a GPU node (srun / sbatch).
"""

from __future__ import annotations

import argparse
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

from src.buffer.house_context_pose_buffer import HouseContextPoseBuffer
from src.r2dreamer.adapters.house_points_adapter import VGGTHousePointsPoseObsAdapter
from src.r2dreamer.launch.habitat_setup import make_habitat_env
from src.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor

VGGT_TOTAL_BUDGET = 200_000
VGGT_STATIC_BUDGETS = tuple([8333] * 24)
RENDER_RESOLUTION = 518

STAGES = ["flatten", "mask", "count", "gather", "quantize", "unique", "newpos",
          "rep_gather", "store"]


class TimedBuffer(HouseContextPoseBuffer):
    """HouseContextPoseBuffer whose ``add`` records per-stage wall time (ms)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stage_ms: dict[str, list[float]] = {s: [] for s in STAGES}
        self.n_active: list[int] = []
        self.n_unique: list[int] = []

    def add_timed(self, vggt_output, observation) -> None:
        rec = {s: 0.0 for s in STAGES}

        t = time.perf_counter()
        flat_points, flat_rgb, confidence_flat = self._flatten_aligned_inputs(
            vggt_output, observation
        )
        flat_points.block_until_ready()
        confidence_flat.block_until_ready()
        rec["flatten"] = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        active_mask = self._high_confidence_finite_mask(flat_points, confidence_flat)
        active_mask.block_until_ready()
        rec["mask"] = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        n_active = int(jnp.sum(active_mask))
        rec["count"] = (time.perf_counter() - t) * 1e3
        self.n_active.append(n_active)

        if n_active == 0:
            self.n_unique.append(0)
            for s in STAGES:
                self.stage_ms[s].append(rec[s])
            return

        t = time.perf_counter()
        valid_points = flat_points[active_mask]
        valid_rgb = flat_rgb[active_mask]
        valid_points.block_until_ready()
        valid_rgb.block_until_ready()
        rec["gather"] = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        scaled = jnp.asarray(valid_points) / self.voxel_size_m
        quantized = jnp.floor(scaled).astype(jnp.int32)
        quantized.block_until_ready()
        rec["quantize"] = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        unique_voxels, first_indices = jnp.unique(
            quantized, axis=0, return_index=True
        )
        unique_voxels.block_until_ready()
        first_indices.block_until_ready()
        rec["unique"] = (time.perf_counter() - t) * 1e3
        self.n_unique.append(int(unique_voxels.shape[0]))

        t = time.perf_counter()
        new_positions = self._new_voxel_positions(unique_voxels)
        rec["newpos"] = (time.perf_counter() - t) * 1e3

        if new_positions:
            # rep_gather: select the K representative points/colors (variable K)
            t = time.perf_counter()
            rep = first_indices[jnp.asarray(new_positions, dtype=jnp.int32)]
            sel_pts = valid_points[rep]
            sel_rgb = valid_rgb[rep]
            sel_pts.block_until_ready()
            sel_rgb.block_until_ready()
            rec["rep_gather"] = (time.perf_counter() - t) * 1e3

            # store: append them into the buffer (growing jnp.concatenate)
            t = time.perf_counter()
            pts = self._append_points(sel_pts, sel_rgb)
            pts.block_until_ready()
            rec["store"] = (time.perf_counter() - t) * 1e3

        for s in STAGES:
            self.stage_ms[s].append(rec[s])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--curriculum", type=str, default="L1")
    return p.parse_args()


def med(xs: list[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def main() -> None:
    args = parse_args()
    print(f"jax backend: {jax.default_backend()}  devices: {jax.devices()}")
    print(f"config: curriculum={args.curriculum}  render={RENDER_RESOLUTION}  "
          f"steps={args.steps} (warmup {args.warmup})  seed={args.seed}\n")

    env = make_habitat_env(
        curriculum=args.curriculum, render_resolution=RENDER_RESOLUTION, seed=args.seed
    )
    extractor = JAXVGGTFeatureExtractor(
        total_budget=VGGT_TOTAL_BUDGET,
        budgets_static=VGGT_STATIC_BUDGETS,
        compute_heads=True,
    )
    # Adapter reused ONLY for its production stride helper + config; the timed
    # buffer replaces the adapter's internal buffer so we own the add() timing.
    adapter = VGGTHousePointsPoseObsAdapter(extractor)
    buffer = TimedBuffer(
        confidence_score=adapter._confidence_score,
        scene_id="bench",
        voxel_size_m=adapter._voxel_size_m,
    )
    print(f"adapter: conf>={adapter._confidence_score}  voxel={adapter._voxel_size_m} m  "
          f"max_input_points={adapter._max_input_points}\n")

    rng = np.random.default_rng(args.seed)
    env.reset()
    extractor.reset()

    total_ms: list[float] = []
    print(f"{'step':>4} {'add_total':>10} {'n_active':>8} {'n_uniq':>7} "
          + " ".join(f"{s:>9}" for s in STAGES))
    for step in range(args.steps):
        action = int(rng.integers(0, env.num_actions))
        obs = env.step(action)
        out = extractor.extract(obs)
        out.world_points.block_until_ready()
        out.confidence.block_until_ready()

        buffer_out, buffer_obs = adapter._subsampled_buffer_input(out, obs)
        before = {s: len(buffer.stage_ms[s]) for s in STAGES}
        buffer.add_timed(buffer_out, buffer_obs)
        row = {s: buffer.stage_ms[s][-1] for s in STAGES}
        add_total = sum(row.values())

        na = buffer.n_active[-1]
        nu = buffer.n_unique[-1]
        print(f"{step:4d} {add_total:9.2f}m {na:8d} {nu:7d} "
              + " ".join(f"{row[s]:8.2f}m" for s in STAGES))
        if step >= args.warmup:
            total_ms.append(add_total)
        if obs.done:
            env.reset()
            extractor.reset()

    lo = args.warmup
    print(f"\n=== per-stage medians (excluding first {args.warmup} warmup steps) ===")
    add_med = med(total_ms)
    for s in STAGES:
        m = med(buffer.stage_ms[s][lo:])
        share = m / add_med * 100 if add_med else 0.0
        print(f"  {s:9s}: median {m:8.2f} ms  ({share:4.0f}% of add)  "
              f"max {max(buffer.stage_ms[s][lo:]):8.2f}")
    print(f"  {'-' * 40}")
    print(f"  {'add total':9s}: median {add_med:8.2f} ms")
    print(f"\n  n_active median {int(med([float(x) for x in buffer.n_active[lo:]]))}  "
          f"n_unique median {int(med([float(x) for x in buffer.n_unique[lo:]]))}")


if __name__ == "__main__":
    main()
