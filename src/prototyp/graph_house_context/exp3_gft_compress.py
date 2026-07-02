"""Experiment 3: block-wise Graph Fourier Transform compression of RGB.

Implements the graph transform coding pipeline of "Graph Spectral Image
Processing" chapter 5, section 5.2: geometry is assumed coded separately
(voxel/octree side of the live buffer); the color signal is decorrelated by
the GFT of a per-block graph Laplacian built from geometry alone. Because a
single eigendecomposition of the full house Laplacian is intractable
(O(N^3)), the cloud is partitioned into voxel blocks and each block is
transformed independently — the divide-and-conquer both chapters prescribe.

Reports PSNR vs kept-coefficient fraction (rate-distortion) for two modes:
``lowfreq`` (keep leading low-frequency rows; kept indices are implicit — an
honest codec) and ``energy`` (keep highest-energy rows; oracle bound needing
index side information).

Run (CPU login node):
    JAX_PLATFORMS=cpu python -m src.prototyp.graph_house_context.exp3_gft_compress
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

from src.prototype_helpers.graph_ops import (
    block_gft,
    group_indices_by_block,
    truncate_coeffs,
    voxel_block_keys,
)
from src.prototype_helpers.graph_metrics import rgb_psnr
from src.prototype_helpers.ply_io import load_ply_xyzrgb, save_ply_xyzrgb

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLY = (
    REPO_ROOT
    / "output/bench/house_context_50steps/bench_50steps_full_1cm/step_00000_context.ply"
)
DEFAULT_OUT_DIR = REPO_ROOT / "outputs/prototype/graph_house_context/exp3"

COEFF_BYTES = 2  # bfloat16 per stored spectral coefficient
INDEX_BYTES = 2  # uint16 per kept-row index (energy-mode side information)
RAW_RGB_BYTES_PER_POINT = 3  # uint8 R, G, B


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", type=Path, default=DEFAULT_PLY)
    parser.add_argument("--block-size", type=float, default=0.5, help="meters")
    parser.add_argument("--block-k", type=int, default=8)
    parser.add_argument(
        "--max-block-points",
        type=int,
        default=3000,
        help="split larger blocks into chunks to bound eigh cost",
    )
    parser.add_argument(
        "--keep-fractions",
        type=lambda text: [float(value) for value in text.split(",")],
        default=[0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
    )
    parser.add_argument(
        "--export-fractions",
        type=lambda text: [float(value) for value in text.split(",")],
        default=[0.05, 0.2],
        help="lowfreq reconstructions written as PLY for these fractions",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def chunked_blocks(
    blocks: list[np.ndarray], max_block_points: int
) -> list[np.ndarray]:
    """Split oversized blocks so the largest eigh stays bounded."""
    chunked: list[np.ndarray] = []
    for indices in blocks:
        if indices.size <= max_block_points:
            chunked.append(indices)
        else:
            parts = int(np.ceil(indices.size / max_block_points))
            chunked.extend(np.array_split(indices, parts))
    return chunked


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    xyz, rgb = load_ply_xyzrgb(args.ply)
    num_points = xyz.shape[0]
    print(f"loaded {num_points} points from {args.ply}")

    keys = voxel_block_keys(xyz, args.block_size)
    blocks = chunked_blocks(group_indices_by_block(keys), args.max_block_points)
    block_sizes = np.array([indices.size for indices in blocks])
    print(
        f"{len(blocks)} blocks at {args.block_size} m "
        f"(sizes min {block_sizes.min()}, median {int(np.median(block_sizes))}, "
        f"max {block_sizes.max()})"
    )

    xyz_np = np.asarray(xyz)
    rgb01 = jnp.asarray(rgb, dtype=jnp.float32) / 255.0
    rgb01_np = np.asarray(rgb01)

    # Export fractions must be transformed too, or their PLYs would stay the
    # all-zeros init; fold them into the lowfreq sweep.
    lowfreq_fractions = sorted(set(args.keep_fractions) | set(args.export_fractions))
    variants = [("lowfreq", frac) for frac in lowfreq_fractions] + [
        ("energy", frac) for frac in args.keep_fractions if frac < 1.0
    ]
    squared_error = {variant: 0.0 for variant in variants}
    attribute_bytes = {variant: 0 for variant in variants}
    export_rgb = {
        frac: np.zeros((num_points, 3), dtype=np.float32)
        for frac in args.export_fractions
    }

    start = time.perf_counter()
    for block_number, indices in enumerate(blocks):
        block_rgb = rgb01_np[indices]
        if indices.size < 2:
            # Too small to transform: costs raw uint8 bytes, reconstructs exactly.
            for variant in variants:
                attribute_bytes[variant] += indices.size * RAW_RGB_BYTES_PER_POINT
            for frac in args.export_fractions:
                export_rgb[frac][indices] = block_rgb
            continue

        _, basis, coeffs = block_gft(
            jnp.asarray(xyz_np[indices]),
            jnp.asarray(block_rgb),
            k=args.block_k,
        )
        for variant in variants:
            mode, frac = variant
            truncated, kept = truncate_coeffs(coeffs, frac, mode=mode)
            reconstruction = np.asarray(basis @ truncated)
            squared_error[variant] += float(
                np.sum((reconstruction - block_rgb) ** 2)
            )
            attribute_bytes[variant] += kept * 3 * COEFF_BYTES
            if mode == "energy":
                attribute_bytes[variant] += kept * INDEX_BYTES
            if mode == "lowfreq" and frac in export_rgb:
                export_rgb[frac][indices] = reconstruction
        if (block_number + 1) % 200 == 0:
            print(f"  {block_number + 1}/{len(blocks)} blocks")
    transform_seconds = time.perf_counter() - start
    print(f"transformed all blocks in {transform_seconds:.1f}s")

    raw_bytes = num_points * RAW_RGB_BYTES_PER_POINT
    results = []
    for variant in variants:
        mode, frac = variant
        mse_255 = squared_error[variant] * 255.0**2 / (num_points * 3)
        psnr = float("inf") if mse_255 == 0.0 else 10.0 * np.log10(255.0**2 / mse_255)
        results.append(
            {
                "mode": mode,
                "keep_fraction": frac,
                "psnr_db": psnr,
                "attribute_bytes": attribute_bytes[variant],
                "attribute_bits_per_point": attribute_bytes[variant] * 8 / num_points,
                "compression_vs_raw_rgb": raw_bytes / max(attribute_bytes[variant], 1),
            }
        )
        print(
            f"{mode:>8} keep={frac:>5.2f}: PSNR {psnr:6.2f} dB, "
            f"{attribute_bytes[variant] / 1e3:8.1f} kB "
            f"({raw_bytes / max(attribute_bytes[variant], 1):.2f}x vs raw rgb)"
        )

    metrics = {
        "ply": str(args.ply),
        "num_points": num_points,
        "block_size_m": args.block_size,
        "block_k": args.block_k,
        "max_block_points": args.max_block_points,
        "num_blocks": len(blocks),
        "block_size_stats": {
            "min": int(block_sizes.min()),
            "median": int(np.median(block_sizes)),
            "max": int(block_sizes.max()),
        },
        "transform_seconds": transform_seconds,
        "raw_rgb_bytes": raw_bytes,
        "results": results,
    }
    (args.out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    save_block_histogram(block_sizes, args.out_dir / "block_hist.png")
    plot_rd_curves(results, raw_bytes, args.out_dir / "rd_curve.png")

    for frac in args.export_fractions:
        reconstruction_uint8 = jnp.asarray(
            np.rint(np.clip(export_rgb[frac], 0.0, 1.0) * 255.0), dtype=jnp.uint8
        )
        save_ply_xyzrgb(
            args.out_dir / f"recon_p{frac:g}.ply", xyz, reconstruction_uint8
        )
    print(f"outputs in {args.out_dir}")


def save_block_histogram(block_sizes: np.ndarray, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.hist(block_sizes, bins=60)
    axis.set_xlabel("points per block")
    axis.set_ylabel("count")
    axis.set_title("Voxel-block occupancy")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def plot_rd_curves(results: list[dict], raw_bytes: int, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for mode, marker in (("lowfreq", "o"), ("energy", "s")):
        rows = sorted(
            (row for row in results if row["mode"] == mode),
            key=lambda row: row["keep_fraction"],
        )
        fractions = [row["keep_fraction"] for row in rows]
        psnrs = [row["psnr_db"] for row in rows]
        kilobytes = [row["attribute_bytes"] / 1e3 for row in rows]
        axes[0].plot(fractions, psnrs, marker=marker, label=mode)
        axes[1].plot(kilobytes, psnrs, marker=marker, label=mode)
    axes[0].set_xlabel("kept coefficient fraction")
    axes[0].set_ylabel("PSNR [dB]")
    axes[0].set_xscale("log")
    axes[1].set_xlabel("attribute size [kB]")
    axes[1].set_ylabel("PSNR [dB]")
    axes[1].axvline(raw_bytes / 1e3, linestyle="--", color="gray", label="raw rgb")
    for axis in axes:
        axis.legend()
        axis.set_title("Block-GFT rate-distortion")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


if __name__ == "__main__":
    main()
