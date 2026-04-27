"""Build similarity matrices C1 (spatial probe-error) and C2 (temporal T×T).

For each holdout episode (ep1, ep7) in viz-pair-a:
  - C2 temporal: cosine(world_points_i, world_points_j) and cosine(feat_i, feat_j)
                 and cosine(deter_i, deter_j) → 3 T×T matrices + 2 diffs.
                 Tells us whether Dreamer agrees with VGGT about which frames look alike.
  - C1 spatial: per-step probe-prediction error (RMSE) per patch, projected onto
                the 37×37 VGGT grid. Shows WHERE in 3D space the latent fails to
                reconstruct geometry.

Saves:
  - similarity.npz (all matrices) per episode
  - similarity.png (composite C2 figure: S_VGGT | S_feat | S_deter | diff_feat | diff_deter)
  - probe_error_timeline.png per episode
  - SUMMARY.md with quantitative agreement stats
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _cosine_matrix(X: np.ndarray) -> np.ndarray:
    """X: (T, D) -> (T, T) cosine similarity."""
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return Xn @ Xn.T


def _load_episode(ep_dir: Path):
    steps = sorted(p for p in ep_dir.iterdir() if p.name.startswith("step_") and p.suffix == ".npz")
    feat, deter, wp = [], [], []
    for s in steps:
        d = np.load(s)
        feat.append(d["feat"])
        deter.append(d["deter"])
        wp.append(d["world_points"].reshape(-1))
    return np.stack(feat), np.stack(deter), np.stack(wp), len(steps)


def _agreement_stats(S_v: np.ndarray, S_d: np.ndarray) -> dict:
    """Compute scalar agreement metrics between two T×T similarity matrices."""
    T = S_v.shape[0]
    iu = np.triu_indices(T, k=1)
    v_off = S_v[iu]
    d_off = S_d[iu]
    diff = S_v - S_d
    return {
        "frobenius_diff": float(np.linalg.norm(diff)),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "pearson_off": float(np.corrcoef(v_off, d_off)[0, 1]),
        "spearman_off": float(_spearman(v_off, d_off)),
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _render_c2(out_path: Path, ep_idx: int, S_v, S_f, S_d, T):
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
    ims = []
    for ax, mat, title, vmin, vmax, cmap in [
        (axes[0], S_v, f"S_VGGT (T={T})", -1, 1, "RdBu_r"),
        (axes[1], S_f, "S_DREAMER feat",   -1, 1, "RdBu_r"),
        (axes[2], S_d, "S_DREAMER deter",  -1, 1, "RdBu_r"),
        (axes[3], S_v - S_f, "diff (VGGT − feat)",  -0.5, 0.5, "coolwarm"),
        (axes[4], S_v - S_d, "diff (VGGT − deter)", -0.5, 0.5, "coolwarm"),
    ]:
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("frame j"); ax.set_ylabel("frame i")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ims.append(im)
    fig.suptitle(f"Episode {ep_idx}: temporal similarity matrices", fontsize=12)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _render_probe_timeline(out_path: Path, ep_idx: int, mse_feat, mse_deter, mse_stoch):
    T = len(mse_feat)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.sqrt(mse_feat),  label="probe_feat",  lw=1.5)
    ax.plot(np.sqrt(mse_deter), label="probe_deter", lw=1.5)
    ax.plot(np.sqrt(mse_stoch), label="probe_stoch", lw=1.5)
    ax.set_xlabel("step")
    ax.set_ylabel("RMSE per step (m)")
    ax.set_title(f"Episode {ep_idx}: probe reconstruction error over trajectory")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dump_dir", required=True)
    p.add_argument("--probe_dir", required=True)
    p.add_argument("--episodes", default="1,7", help="csv of episode indices")
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    dump_dir = Path(args.dump_dir)
    probe_dir = Path(args.probe_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    eps = [int(x) for x in args.episodes.split(",")]

    summary = {}
    for ep_idx in eps:
        print(f"\n=== Episode {ep_idx} ===")
        ep_out = out_root / f"ep_{ep_idx:03d}"
        ep_out.mkdir(exist_ok=True)
        ep_dir = dump_dir / f"episode_{ep_idx:03d}"
        feat, deter, wp, T = _load_episode(ep_dir)
        print(f"  loaded T={T} steps")

        S_v = _cosine_matrix(wp)
        S_f = _cosine_matrix(feat)
        S_d = _cosine_matrix(deter)
        stats_feat  = _agreement_stats(S_v, S_f)
        stats_deter = _agreement_stats(S_v, S_d)

        print(f"  feat  vs VGGT: pearson={stats_feat['pearson_off']:.3f}  mean|diff|={stats_feat['mean_abs_diff']:.3f}")
        print(f"  deter vs VGGT: pearson={stats_deter['pearson_off']:.3f}  mean|diff|={stats_deter['mean_abs_diff']:.3f}")

        np.savez_compressed(
            ep_out / "similarity.npz",
            S_VGGT=S_v, S_feat=S_f, S_deter=S_d,
            diff_feat=S_v - S_f, diff_deter=S_v - S_d,
        )
        _render_c2(ep_out / "similarity.png", ep_idx, S_v, S_f, S_d, T)

        # Probe error timeline (uses fit_probes.py output)
        pred = np.load(probe_dir / f"predictions_ep{ep_idx:03d}.npz")
        _render_probe_timeline(
            ep_out / "probe_error_timeline.png",
            ep_idx, pred["mse_feat"], pred["mse_deter"], pred["mse_stoch"],
        )

        summary[ep_idx] = {
            "T": T,
            "feat_vs_vggt": stats_feat,
            "deter_vs_vggt": stats_deter,
            "probe_rmse_mean": {
                "feat":  float(np.sqrt(pred["mse_feat"]).mean()),
                "deter": float(np.sqrt(pred["mse_deter"]).mean()),
                "stoch": float(np.sqrt(pred["mse_stoch"]).mean()),
            },
        }

    # Write SUMMARY.md
    with open(out_root / "SUMMARY.md", "w") as f:
        f.write("# Similarity-matrix summary (C1+C2)\n\n")
        f.write(f"- dump_dir: `{dump_dir}`\n- probe_dir: `{probe_dir}`\n- episodes: {eps}\n\n")
        f.write("## Pearson correlation of off-diagonal cosine entries (Dreamer vs VGGT)\n\n")
        f.write("Higher = Dreamer agrees with VGGT about which frame-pairs are similar.\n\n")
        f.write("| ep | T | feat-pearson | deter-pearson | mean|diff_feat| | mean|diff_deter| |\n")
        f.write("|---|---|---|---|---|---|\n")
        for ep_idx in eps:
            s = summary[ep_idx]
            f.write(
                f"| {ep_idx} | {s['T']} "
                f"| {s['feat_vs_vggt']['pearson_off']:.3f} "
                f"| {s['deter_vs_vggt']['pearson_off']:.3f} "
                f"| {s['feat_vs_vggt']['mean_abs_diff']:.3f} "
                f"| {s['deter_vs_vggt']['mean_abs_diff']:.3f} |\n"
            )
        f.write("\n## Probe-error mean RMSE per episode (m)\n\n")
        f.write("| ep | feat RMSE | deter RMSE | stoch RMSE |\n|---|---|---|---|\n")
        for ep_idx in eps:
            r = summary[ep_idx]["probe_rmse_mean"]
            f.write(f"| {ep_idx} | {r['feat']:.3f} | {r['deter']:.3f} | {r['stoch']:.3f} |\n")
        f.write("\n## Files\n\n")
        for ep_idx in eps:
            f.write(f"- `ep_{ep_idx:03d}/similarity.png` — C2 temporal panel\n")
            f.write(f"- `ep_{ep_idx:03d}/similarity.npz` — raw matrices\n")
            f.write(f"- `ep_{ep_idx:03d}/probe_error_timeline.png` — C1 RMSE trajectory\n")

    print(f"\nWrote SUMMARY.md and per-episode artifacts to {out_root}/")


if __name__ == "__main__":
    main()
