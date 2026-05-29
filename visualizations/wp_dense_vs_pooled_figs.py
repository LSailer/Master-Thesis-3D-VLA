"""Appendix figures for 3D-48: dense (518x518x3) vs pooled (37x37x3) world points.

Reads the npz produced by scripts/vggt/dump_wp_dense_vs_pooled.py and renders
a set of PNGs that make the 14x14 average-pooling loss visually obvious.

Usage:
  .venv/bin/python -m visualizations.wp_dense_vs_pooled_figs \
      --npz output/3d48/wp_dense_vs_pooled.npz \
      --outdir docs/images/3d48
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PATCH = 14
GRID = 37
IMG = 518


def _pct_norm(a: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Per-channel percentile min-max normalise to [0, 1] for display."""
    out = np.empty_like(a, dtype=np.float32)
    for c in range(a.shape[-1]):
        ch = a[..., c]
        p_lo, p_hi = np.percentile(ch, [lo, hi])
        out[..., c] = np.clip((ch - p_lo) / (p_hi - p_lo + 1e-9), 0, 1)
    return out


def fig_xyz_image(dense, pooled, outdir: Path) -> None:
    """XYZ map as a false-colour image: dense vs pooled (nearest-upsampled)."""
    dense_rgb = _pct_norm(dense)
    pooled_rgb = _pct_norm(pooled)
    pooled_up = np.repeat(np.repeat(pooled_rgb, PATCH, axis=0), PATCH, axis=1)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
    ax[0].imshow(dense_rgb)
    ax[0].set_title(f"Dense pts3d  {dense.shape[0]}x{dense.shape[1]}x3\n"
                    f"{dense.shape[0] * dense.shape[1]:,} points (1 / pixel)")
    ax[1].imshow(pooled_up, interpolation="nearest")
    ax[1].set_title(f"Pooled world_points {GRID}x{GRID}x3\n"
                    f"{GRID * GRID:,} points, shown upsampled (blocky)")
    ax[2].imshow(pooled_rgb, interpolation="nearest")
    ax[2].set_title(f"Pooled, native {GRID}x{GRID}\n(what we feed R2Dreamer)")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("World-point map (XYZ -> RGB) — 14x14 average pooling loss",
                 fontsize=13, y=1.06)
    fig.tight_layout()
    fig.savefig(outdir / "fig_xyz_image.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_depth(dense, pooled, outdir: Path) -> None:
    """Depth (Z channel) heatmap: dense vs pooled-upsampled."""
    z_d = dense[..., 2]
    z_p = pooled[..., 2]
    z_p_up = np.repeat(np.repeat(z_p, PATCH, axis=0), PATCH, axis=1)
    vmin, vmax = np.percentile(z_d, [2, 98])

    fig, ax = plt.subplots(1, 2, figsize=(11, 5.2))
    im0 = ax[0].imshow(z_d, cmap="viridis", vmin=vmin, vmax=vmax)
    ax[0].set_title(f"Dense depth (Z)  {IMG}x{IMG}")
    ax[1].imshow(z_p_up, cmap="viridis", vmin=vmin, vmax=vmax,
                 interpolation="nearest")
    ax[1].set_title(f"Pooled depth (Z)  {GRID}x{GRID} -> upsampled")
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im0, ax=ax, fraction=0.025, pad=0.02, label="Z (depth)")
    fig.suptitle("Depth detail lost to 14x14 pooling", fontsize=13)
    fig.savefig(outdir / "fig_depth.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_grid_overlay(rgb_chw, outdir: Path) -> None:
    """Input RGB with the 37x37 patch grid overlaid; one cell highlighted."""
    rgb = np.transpose(rgb_chw, (1, 2, 0))  # HWC uint8
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.imshow(rgb)
    for i in range(GRID + 1):
        ax.axhline(i * PATCH - 0.5, color="white", lw=0.3, alpha=0.5)
        ax.axvline(i * PATCH - 0.5, color="white", lw=0.3, alpha=0.5)
    # Highlight one cell (centre-ish) — all 14x14 = 196 pixels -> 1 point.
    r, c = 18, 18
    ax.add_patch(plt.Rectangle((c * PATCH - 0.5, r * PATCH - 0.5), PATCH, PATCH,
                               fill=False, edgecolor="red", lw=2.0))
    ax.set_title(f"Input RGB {IMG}x{IMG} with {GRID}x{GRID} patch grid\n"
                 f"each red 14x14 cell (196 px) -> ONE pooled 3D point")
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(outdir / "fig_grid_overlay.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_pointcloud(dense, pooled, outdir: Path) -> None:
    """Side-by-side 3D scatter, coloured by depth, shared view."""
    d = dense.reshape(-1, 3)
    p = pooled.reshape(-1, 3)
    stride = max(1, d.shape[0] // 40000)
    ds = d[::stride]

    fig = plt.figure(figsize=(13, 6.2))
    for k, (pts, title) in enumerate([
        (ds, f"Dense (~{ds.shape[0]:,} of {d.shape[0]:,} pts shown)"),
        (p, f"Pooled ({p.shape[0]:,} pts)"),
    ]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2],
                   cmap="viridis", s=(1 if k == 0 else 8), alpha=0.6)
        ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.view_init(elev=-65, azim=-90)
    fig.suptitle("Reconstructed point cloud: dense vs pooled", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "fig_pointcloud.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_block_zoom(dense, pooled, outdir: Path) -> None:
    """Zoom one 14x14 dense block and show the single pooled value it becomes."""
    r, c = 18, 18
    block = dense[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH, :]
    mean_pt = block.reshape(-1, 3).mean(0)
    pooled_pt = pooled[r, c]

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.6))
    ax[0].imshow(_pct_norm(block), interpolation="nearest")
    ax[0].set_title(f"Dense 14x14 block (196 points)\ncell ({r},{c})")
    ax[0].set_xticks([]); ax[0].set_yticks([])
    txt = (f"avg of block  = [{mean_pt[0]:+.3f}, {mean_pt[1]:+.3f}, {mean_pt[2]:+.3f}]\n"
           f"pooled value  = [{pooled_pt[0]:+.3f}, {pooled_pt[1]:+.3f}, {pooled_pt[2]:+.3f}]\n\n"
           f"196 distinct 3D points  ->  1 point\n(adaptive_avg_pool, identical to the mean)")
    ax[1].axis("off")
    ax[1].text(0.02, 0.5, txt, family="monospace", fontsize=11, va="center")
    fig.suptitle("What pooling does to a single patch cell", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "fig_block_zoom.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", required=True, type=Path)
    p.add_argument("--outdir", required=True, type=Path)
    args = p.parse_args()

    d = np.load(args.npz)
    dense = d["dense_world_points"]      # (518, 518, 3)
    pooled = d["world_points"]           # (37, 37, 3)
    rgb = d["input_rgb"]                 # (3, 518, 518) uint8
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"dense {dense.shape}  pooled {pooled.shape}  rgb {rgb.shape}")
    fig_xyz_image(dense, pooled, args.outdir)
    fig_depth(dense, pooled, args.outdir)
    fig_grid_overlay(rgb, args.outdir)
    fig_pointcloud(dense, pooled, args.outdir)
    fig_block_zoom(dense, pooled, args.outdir)
    print(f"Wrote figures to {args.outdir}")


if __name__ == "__main__":
    main()
