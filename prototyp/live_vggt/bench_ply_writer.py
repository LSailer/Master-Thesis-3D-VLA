"""Benchmark: old Open3D PLY writer vs the new direct-NumPy writer.

Times both implementations on synthetic XYZRGB clouds (bfloat16 device
coordinates + uint8 colours, mimicking the VGGT accumulation path) and
prints a markdown table of write seconds and file MB. Files go to
``outputs/prototype/live_vggt/bench/`` and are deleted after timing.

Run on the login node:
    JAX_PLATFORMS=cpu .venv/bin/python prototyp/live_vggt/bench_ply_writer.py \
        --sizes 1000000 10000000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp

# File writing and Open3D interop are host-side; NumPy marks the boundary.
import numpy as np

from src.shared.ply_io import flattened_xyz_rgb, write_world_points_ply


def write_world_points_ply_open3d(path: Path, xyz: jnp.ndarray, rgb: jnp.ndarray) -> None:
    """Old Open3D-backed writer, inlined verbatim for comparison.

    Args:
      path: Output ``.ply`` path.
      xyz: ``(..., 3)`` float vertex coordinates.
      rgb: ``(..., 3)`` uint8 or float ``[0, 1]`` vertex colours.

    Returns:
      None.
    """
    import open3d as o3d

    xyz_flat, rgb_flat = flattened_xyz_rgb(xyz, rgb)
    pcd = o3d.geometry.PointCloud()
    # Open3D's Vector3dVector wraps Eigen float64 vectors and requires host
    # memory: np.asarray performs the explicit device -> host transfer.
    pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz_flat, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(rgb_flat, dtype=np.float64))
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)


def synthetic_cloud(n: int, seed: int = 0) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate a synthetic point cloud shaped like the VGGT accumulation.

    Args:
      n: Number of points.
      seed: PRNG seed.

    Returns:
      Tuple ``(xyz, rgb)``: ``(n, 3)`` bfloat16 coordinates in roughly
      ``[-5, 5]`` and ``(n, 3)`` uint8 colours.
    """
    kx, kc = jax.random.split(jax.random.PRNGKey(seed))
    xyz = jax.random.uniform(
        kx, (n, 3), dtype=jnp.float32, minval=-5.0, maxval=5.0
    ).astype(jnp.bfloat16)
    rgb = jax.random.randint(kc, (n, 3), 0, 256, dtype=jnp.int32).astype(jnp.uint8)
    return xyz, rgb


def _timed_write(writer, path: Path, xyz: jnp.ndarray, rgb: jnp.ndarray) -> tuple[float, float]:
    """Time one write and return (seconds, MB written); deletes the file.

    Args:
      writer: Callable ``writer(path, xyz, rgb)``.
      path: Output file path.
      xyz: ``(n, 3)`` coordinates.
      rgb: ``(n, 3)`` colours.

    Returns:
      Tuple ``(seconds, megabytes)`` for the write.
    """
    t0 = time.perf_counter()
    writer(path, xyz, rgb)
    seconds = time.perf_counter() - t0
    mb = path.stat().st_size / 1e6
    path.unlink()
    return seconds, mb


def main() -> None:
    """Run the writer comparison and print a markdown table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1_000_000, 10_000_000, 50_000_000],
        help="Point counts to benchmark.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/prototype/live_vggt/bench",
        help="Scratch directory for the timed files (deleted after timing).",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"jax backend: {jax.default_backend()}")

    # Warmup at a small size so import/JIT costs stay out of the timings.
    xyz_w, rgb_w = synthetic_cloud(10_000)
    _timed_write(write_world_points_ply, out_dir / "warm_new.ply", xyz_w, rgb_w)
    _timed_write(write_world_points_ply_open3d, out_dir / "warm_old.ply", xyz_w, rgb_w)

    rows = []
    for n in args.sizes:
        xyz, rgb = synthetic_cloud(n)
        jax.block_until_ready(xyz)
        old_s, old_mb = _timed_write(
            write_world_points_ply_open3d, out_dir / f"old_{n}.ply", xyz, rgb
        )
        new_s, new_mb = _timed_write(
            write_world_points_ply, out_dir / f"new_{n}.ply", xyz, rgb
        )
        rows.append((n, old_s, old_mb, new_s, new_mb))
        print(f"[bench] n={n:,}: open3d {old_s:.2f}s / numpy {new_s:.2f}s", flush=True)

    print()
    print("| points | open3d (s) | open3d (MB) | numpy (s) | numpy (MB) | speedup |")
    print("|-------:|-----------:|------------:|----------:|-----------:|--------:|")
    for n, old_s, old_mb, new_s, new_mb in rows:
        print(
            f"| {n:,} | {old_s:.2f} | {old_mb:.1f} | {new_s:.2f} | "
            f"{new_mb:.1f} | {old_s / new_s:.1f}x |"
        )


if __name__ == "__main__":
    main()
