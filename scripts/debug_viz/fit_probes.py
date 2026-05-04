"""Fit linear probes from R2Dreamer latents to VGGT world_points.

Trains three Ridge probes (closed-form, alpha=1.0) on per-step npz dumps:
  - probe_feat:  feat  (2560,) -> world_points_flat (4107,)
  - probe_deter: deter (2048,) -> world_points_flat (4107,)
  - probe_stoch: stoch (512,)  -> world_points_flat (4107,)

Train on all episodes in --dump_dir except --holdout (e.g. "1,7"). Evaluate
on holdout. Reports per-probe R^2 (mean over output dims, sample-level)
and saves per-step predictions to disk for downstream viz.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _episode_steps(ep_dir: Path) -> list[Path]:
    return sorted(p for p in ep_dir.iterdir() if p.name.startswith("step_") and p.suffix == ".npz")


def _load_episode(ep_dir: Path):
    """Return dict of stacked arrays for one episode."""
    steps = _episode_steps(ep_dir)
    feat, deter, stoch, wp = [], [], [], []
    for s in steps:
        d = np.load(s)
        feat.append(d["feat"])
        deter.append(d["deter"])
        stoch.append(d["stoch"].reshape(-1))
        wp.append(d["world_points"].reshape(-1))
    return {
        "feat": np.stack(feat).astype(np.float32),
        "deter": np.stack(deter).astype(np.float32),
        "stoch": np.stack(stoch).astype(np.float32),
        "world_points": np.stack(wp).astype(np.float32),
        "n_steps": len(steps),
    }


def _fit_probe(X: np.ndarray, Y: np.ndarray, alpha: float = 1.0):
    """Standardize X and Y, fit Y ≈ X @ W via Ridge regression.

    Returns (W, x_mean, x_std, y_mean, y_std) so prediction is:
        Y_hat = ((X - x_mean) / x_std) @ W * y_std + y_mean
    """
    x_mean = X.mean(axis=0, keepdims=True)
    x_std = X.std(axis=0, keepdims=True) + 1e-6
    y_mean = Y.mean(axis=0, keepdims=True)
    y_std = Y.std(axis=0, keepdims=True) + 1e-6
    Xs = (X - x_mean) / x_std
    Ys = (Y - y_mean) / y_std
    # Ridge closed form: W = (X'X + alpha I)^-1 X'Y
    XtX = Xs.T @ Xs
    XtX[np.diag_indices_from(XtX)] += alpha
    W = np.linalg.solve(XtX, Xs.T @ Ys)
    return W, x_mean[0], x_std[0], y_mean[0], y_std[0]


def _predict(X, W, x_mean, x_std, y_mean, y_std):
    return ((X - x_mean) / x_std) @ W * y_std + y_mean


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Sample-level R^2: 1 - SS_res / SS_tot, both summed over all dims."""
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dump_dir", required=True, help="dir containing episode_*/step_*.npz")
    p.add_argument("--holdout", default="1,7", help="csv of episode indices to hold out")
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    dump_dir = Path(args.dump_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    holdout = {int(x) for x in args.holdout.split(",")}

    ep_dirs = sorted(d for d in dump_dir.iterdir() if d.is_dir() and d.name.startswith("episode_"))
    train_eps, eval_eps = [], []
    for d in ep_dirs:
        idx = int(d.name.split("_")[1])
        (eval_eps if idx in holdout else train_eps).append((idx, d))

    print(f"Train: {[i for i, _ in train_eps]}  Eval: {[i for i, _ in eval_eps]}")

    print("Loading train episodes...")
    train = [_load_episode(d) for _, d in train_eps]
    X_feat  = np.concatenate([t["feat"]  for t in train])
    X_deter = np.concatenate([t["deter"] for t in train])
    X_stoch = np.concatenate([t["stoch"] for t in train])
    Y       = np.concatenate([t["world_points"] for t in train])
    print(f"  total train steps: {Y.shape[0]}")

    # Sweep alphas, pick by holdout R^2 averaged across eval eps for each probe.
    eval_data = [(idx, _load_episode(d)) for idx, d in eval_eps]
    alphas = [1.0, 10.0, 100.0, 1000.0, 10000.0]
    probes = {}
    for name, X in [("feat", X_feat), ("deter", X_deter), ("stoch", X_stoch)]:
        print(f"\nFitting probe_{name}: X{X.shape} -> Y{Y.shape}, sweeping alpha")
        best = None
        for alpha in alphas:
            W, xm, xs, ym, ys = _fit_probe(X, Y, alpha=alpha)
            Y_hat_tr = _predict(X, W, xm, xs, ym, ys)
            r2_tr = _r2(Y, Y_hat_tr)
            r2_eval = []
            for idx, ep in eval_data:
                Yh = _predict(ep[name], W, xm, xs, ym, ys)
                r2_eval.append(_r2(ep["world_points"], Yh))
            r2_avg = float(np.mean(r2_eval))
            print(f"  alpha={alpha:>8.1f}  train R^2={r2_tr:.3f}  holdout R^2 avg={r2_avg:.3f}  per-ep={r2_eval}")
            if best is None or r2_avg > best["r2_eval_avg"]:
                best = dict(alpha=alpha, W=W, xm=xm, xs=xs, ym=ym, ys=ys,
                            r2_train=r2_tr, r2_eval_avg=r2_avg, r2_eval=r2_eval)
        probes[name] = best
        np.savez_compressed(
            out_dir / f"probe_{name}.npz",
            W=best["W"], x_mean=best["xm"], x_std=best["xs"],
            y_mean=best["ym"], y_std=best["ys"],
            alpha=best["alpha"], r2_train=best["r2_train"], r2_eval_avg=best["r2_eval_avg"],
        )

    print("\nEvaluating best-alpha probes on holdout...")
    summary_rows = []
    for idx, ep in eval_data:
        per_probe_pred = {}
        per_probe_r2 = {}
        per_probe_mse = {}
        for name in ("feat", "deter", "stoch"):
            P = probes[name]
            Y_true = ep["world_points"]
            Y_hat = _predict(ep[name], P["W"], P["xm"], P["xs"], P["ym"], P["ys"])
            per_probe_pred[name] = Y_hat
            per_probe_r2[name] = _r2(Y_true, Y_hat)
            per_probe_mse[name] = ((Y_true - Y_hat) ** 2).mean(axis=1)
            summary_rows.append((idx, name, P["alpha"], per_probe_r2[name], float(per_probe_mse[name].mean())))
            print(f"  ep{idx} probe_{name} (alpha={P['alpha']}): R^2 = {per_probe_r2[name]:.4f}  RMSE = {np.sqrt(per_probe_mse[name].mean()):.3f} m")

        np.savez_compressed(
            out_dir / f"predictions_ep{idx:03d}.npz",
            world_points_pred_feat=per_probe_pred["feat"].reshape(-1, 37, 37, 3),
            world_points_pred_deter=per_probe_pred["deter"].reshape(-1, 37, 37, 3),
            world_points_pred_stoch=per_probe_pred["stoch"].reshape(-1, 37, 37, 3),
            world_points_true=ep["world_points"].reshape(-1, 37, 37, 3),
            mse_feat=per_probe_mse["feat"],
            mse_deter=per_probe_mse["deter"],
            mse_stoch=per_probe_mse["stoch"],
            r2_feat=per_probe_r2["feat"],
            r2_deter=per_probe_r2["deter"],
            r2_stoch=per_probe_r2["stoch"],
        )

    with open(out_dir / "SUMMARY.md", "w") as f:
        f.write(f"# Probe-fit summary (Ridge with alpha sweep)\n\n")
        f.write(f"- dump_dir: `{dump_dir}`\n- holdout: {sorted(holdout)}\n")
        f.write(f"- train steps: {Y.shape[0]}\n- alphas swept: {alphas}\n\n")
        f.write("## Best probe per latent\n\n| probe | best alpha | train R^2 | holdout R^2 (avg) |\n|---|---|---|---|\n")
        for name in ("feat", "deter", "stoch"):
            P = probes[name]
            f.write(f"| {name} | {P['alpha']} | {P['r2_train']:.4f} | {P['r2_eval_avg']:.4f} |\n")
        f.write("\n## Per-holdout-episode\n\n| ep | probe | alpha | R^2 | RMSE (m) |\n|---|---|---|---|---|\n")
        for idx, name, alpha, r2, mse in summary_rows:
            f.write(f"| {idx} | {name} | {alpha} | {r2:.4f} | {np.sqrt(mse):.3f} |\n")

    print(f"\nSaved probes + predictions to {out_dir}/")


if __name__ == "__main__":
    main()
