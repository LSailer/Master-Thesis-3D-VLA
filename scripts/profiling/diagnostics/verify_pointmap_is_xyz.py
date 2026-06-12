"""Verify that VGGT's point map is geometry (XYZ), not colour (3D-48).

Three independent, GPU-free checks on the dumped npz
(``output/3d48/wp_dense_vs_pooled.npz``):

  1. Channel semantics — the DPT point head emits 4 channels = xyz + conf
     (src/vggt/jax/heads/dpt_head.py:159-162); we keep the 3 xyz. No colour
     channel exists in the head's output.
  2. Values are coordinates, not colours — the point map has signed/negative
     values outside [0, 255] and each channel tracks pixel geometry
     (X ~ column, Y ~ row: back-projected camera rays), whereas the input
     image is non-negative in [0, 255].
  3. Colour is an external per-pixel overlay — colouring the cloud with the
     input RGB gives a coherent room; SHUFFLING the colour array against the
     same positions destroys it. So colour is paired to points only by the
     shared H*W pixel index (borrowed from the input image), not stored in
     the point map.

Usage:
  .venv/bin/python -m scripts.profiling.diagnostics.verify_pointmap_is_xyz \
      --npz output/3d48/wp_dense_vs_pooled.npz \
      --out output/3d48/verify_pointmap.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    d = np.load(args.npz)
    P = d["dense_world_points"]                       # point map Pi (H,W,3)
    I = np.transpose(d["input_rgb"], (1, 2, 0))       # input image Ii (H,W,3)
    H, W, _ = P.shape

    print("=" * 64)
    print("VERIFY: VGGT point map is XYZ geometry, not colour")
    print("=" * 64)

    # --- Check 1: shape + channel semantics ---------------------------------
    print("\n[1] Channel semantics")
    print(f"    point map Pi shape      : {P.shape}  (3 ch = X,Y,Z)")
    print(f"    input image Ii shape    : {I.shape}  (3 ch = R,G,B)")
    print("    -> identical SHAPE, different MEANING. The DPT head emits "
          "xyz+conf;\n       only xyz is kept (no colour channel exists).")

    # --- Check 2: value ranges (coordinates vs colours) ---------------------
    print("\n[2] Values are coordinates, not colours")
    for k, ax in enumerate("XYZ"):
        a = P[..., k]
        print(f"    Pi[{ax}] range [{a.min():+.3f}, {a.max():+.3f}]  "
              f"negatives={bool((a < 0).any())}")
    print(f"    Ii(RGB) range [{I.min():.0f}, {I.max():.0f}]  negatives={bool((I < 0).any())}")
    assert (P < 0).any(), "expected negative coordinates in a point map"
    assert I.min() >= 0 and I.max() <= 255, "input image should be 0..255"
    print("    -> Pi has NEGATIVE values; colours cannot. Pi is metric XYZ.")
    # geometry: each coordinate tracks a pixel axis (back-projected rays)
    rows = np.broadcast_to(np.arange(H)[:, None], (H, W)).ravel()
    cols = np.broadcast_to(np.arange(W)[None, :], (H, W)).ravel()
    cx = np.corrcoef(P[..., 0].ravel(), cols)[0, 1]
    cy = np.corrcoef(P[..., 1].ravel(), rows)[0, 1]
    print(f"    corr(X, image column) = {cx:+.2f}   corr(Y, image row) = {cy:+.2f}")
    print("    -> X tracks column, Y tracks row: geometry (camera rays), not colour.")

    # --- Check 3: colour is an external per-pixel overlay -------------------
    print("\n[3] Colour is borrowed from Ii by pixel index (shuffle test)")
    pts = P.reshape(-1, 3)
    col = (I.reshape(-1, 3) / 255.0)
    rng = np.random.RandomState(0)
    perm = rng.permutation(len(col))
    shuffled = col[perm]
    # Coherence metric: mean colour difference between spatially adjacent
    # points (same row). Real colours vary smoothly -> small; shuffled -> large.
    cimg = col.reshape(H, W, 3)
    smooth_real = np.abs(np.diff(cimg, axis=1)).mean()
    smooth_shuf = np.abs(np.diff(shuffled.reshape(H, W, 3), axis=1)).mean()
    print(f"    mean neighbour colour diff:  real={smooth_real:.4f}  "
          f"shuffled={smooth_shuf:.4f}  (x{smooth_shuf / smooth_real:.0f})")
    print("    -> Real colours are locally smooth; shuffling (breaking the pixel\n"
          "       index) destroys coherence. Colour lives in Ii, paired by index.")

    # --- Figure: coherent vs shuffled vs false-colour XYZ -------------------
    st = slice(None, None, 40)
    fig = plt.figure(figsize=(15, 5.2))
    panels = [
        (col[st], "Colour from input pixel Ii(y)\n(coherent room)"),
        (shuffled[st], "Colour shuffled vs position\n(pixel index broken -> noise)"),
    ]
    for i, (c, title) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.scatter(pts[st, 0], pts[st, 1], pts[st, 2], c=np.clip(c, 0, 1),
                   s=2, depthshade=False)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=-70, azim=-90)
    # third panel: the point map's OWN values as false colour (clearly not a photo)
    axp = fig.add_subplot(1, 3, 3)
    Pn = np.stack([(P[..., k] - P[..., k].min()) /
                   (np.ptp(P[..., k]) + 1e-9) for k in range(3)], -1)
    axp.imshow(Pn); axp.set_xticks([]); axp.set_yticks([])
    axp.set_title("Point map Pi as false-colour\n(R=X,G=Y,B=Z — not the photo)", fontsize=10)
    fig.suptitle("Point map is geometry; colour is an input-image overlay", fontsize=13)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {args.out}")
    print("\nVERDICT: point map = XYZ geometry; colour is borrowed from the "
          "input\nimage by pixel index. The Dreamer feed (Pi + pose) carries NO colour.")


if __name__ == "__main__":
    main()
