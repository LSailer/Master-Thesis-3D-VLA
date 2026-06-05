"""Cross-run analysis for the WP-resolution / encoder ablation on L1 ObjectNav.

Compares the prod runs launched for 3D-50/51/52:

  CONTROLLED (current code, identical config except WP pool resolution):
    * WP 37x37 MLP  (vggt,        obs 4116,  depth-3 MLP)  -- run 4887404
    * WP 64x64 MLP  (vggt_wp_cp_64, obs 12297, depth-3 MLP) -- run 4888735 (partial, hung @ ~50k)

  REFERENCE (historical, different branch/date -- context only, drawn dashed):
    * WP 37x37 linear (vggt, obs 4116, linear readout) -- run 4216462
    * RGB CNN 64x64   (cnn,  obs 3x64x64)               -- run 4367942

Because each run uses a single seed (seed = SLURM_JOB_ID), there are no error
bars; treat single-run differences with care. The controlled pair only overlaps
on [0, ~50k] env steps, so the resolution comparison is read in that window.

Outputs PNGs to output/analysis/wp-cp-resolution/ and prints a summary table.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.facecolor": "#0f0f13",
    "axes.facecolor": "#1a1a24",
    "axes.edgecolor": "#2d2d3d",
    "axes.labelcolor": "#e2e8f0",
    "text.color": "#e2e8f0",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "grid.color": "#2d2d3d",
    "font.family": "sans-serif",
    "font.size": 11,
})

OUTPUT_DIR = "output/analysis/wp-cp-resolution"

# label -> (csv, color, linestyle, controlled?)
RUNS = {
    "WP 37² MLP (4887404)": (
        "output/r2dreamer-curriculum-l1-vggt-wp-cp-mlp/run-4887404/metrics.csv",
        "#6366f1", "-", True),
    "WP 64² MLP (4888735, partial)": (
        "output/r2dreamer-curriculum-l1-vggt-wp-cp-64/run-4888735/metrics.csv",
        "#34d399", "-", True),
    "WP 37² linear [ref]": (
        "output/r2dreamer-curriculum-l1-vggt/run-4216462/metrics.csv",
        "#22d3ee", "--", False),
    "RGB CNN 64² [ref] (lhgoxh0y)": (
        "output/r2dreamer-curriculum-l1/run-4194043/metrics.csv",
        "#fbbf24", "--", False),
}

# Common step ceiling for the controlled pair (where 64x64 stopped).
COMMON_STEP = 50_000


def load_run(csv):
    df = pd.read_csv(csv)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    return df.dropna(subset=["step"])


def series(df, name, rolling=0):
    sub = df[df["metric"] == name].sort_values("step").dropna(subset=["value"])
    if sub.empty:
        return None
    s = sub["value"].to_numpy()
    if rolling > 0 and len(s) > rolling:
        s = pd.Series(s).rolling(rolling, min_periods=1).mean().to_numpy()
    return sub["step"].to_numpy(), s


def value_at(df, name, step):
    """Last logged value of `name` at/below `step` (rolling-mean metrics)."""
    sub = df[(df["metric"] == name) & (df["step"] <= step)].dropna(subset=["value"])
    return float(sub.sort_values("step")["value"].iloc[-1]) if not sub.empty else float("nan")


def panel(ax, data, metric, title, ylabel, rolling, scale=1.0, xmax=None):
    for label, (df, color, ls, _ctrl) in data.items():
        r = series(df, metric, rolling=rolling)
        if r is None:
            continue
        x, y = r
        x = x / 1e6
        ax.plot(x, y * scale, color=color, linestyle=ls, linewidth=1.9,
                label=label, alpha=0.95 if ls == "-" else 0.7)
    ax.set_title(title, fontweight="bold", fontsize=12)
    ax.set_xlabel("Environment steps (M)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    if xmax is not None:
        ax.set_xlim(0, xmax / 1e6)


def fig_task(data, fname, xmax=None, tag=""):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    panel(axes[0, 0], data, "metrics/sr", "Success rate", "SR", rolling=20, scale=100, xmax=xmax)
    axes[0, 0].set_ylabel("Success rate (%)")
    panel(axes[0, 1], data, "metrics/spl", "SPL", "SPL", rolling=20, xmax=xmax)
    panel(axes[1, 0], data, "metrics/reward", "Mean reward (rolling)", "Reward", rolling=20, xmax=xmax)
    panel(axes[1, 1], data, "metrics/dtg", "Distance-to-goal (lower=better)", "DTG (m)", rolling=20, xmax=xmax)
    if xmax is not None:
        for ax in axes.ravel():
            ax.axvline(COMMON_STEP / 1e6, color="#64748b", linestyle=":", alpha=0.6)
    axes[0, 0].legend(fontsize=9, loc="upper left")
    fig.suptitle(f"L1 ObjectNav -- task metrics{tag}", fontweight="bold", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig_wm(data, fname):
    losses = [("loss/dyn", "Dynamics KL"), ("loss/rep", "Representation KL"),
              ("loss/rew", "Reward prediction"), ("loss/barlow", "Barlow-Twins"),
              ("loss/con", "Contrastive"), ("total_loss", "Total loss")]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    for ax, (m, t) in zip(axes.ravel(), losses):
        panel(ax, data, m, t, "loss", rolling=50)
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle("World-model losses", fontweight="bold", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def fig_health(label, df, fname):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, (m, t, yl, roll) in zip(axes.ravel(), [
            ("loss/policy", "Actor (policy) loss", "loss", 50),
            ("loss/value", "Critic (value) loss", "loss", 50),
            ("latent/posterior_entropy", "Latent entropy (post vs prior)", "nats", 50),
            ("params/encoder_l2", "Encoder weight L2", "L2", 0)]):
        r = series(df, m, rolling=roll)
        if r:
            ax.plot(r[0] / 1e6, r[1], color="#6366f1", linewidth=1.9, label=t)
        if m == "latent/posterior_entropy":
            rp = series(df, "latent/prior_entropy", rolling=roll)
            if rp:
                ax.plot(rp[0] / 1e6, rp[1], color="#fbbf24", linewidth=1.9, label="prior")
            ax.legend(fontsize=9)
        ax.set_title(t, fontweight="bold", fontsize=12)
        ax.set_xlabel("Environment steps (M)")
        ax.set_ylabel(yl)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Run health -- {label}", fontweight="bold", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUTPUT_DIR, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = {}
    for label, (csv, color, ls, ctrl) in RUNS.items():
        if not os.path.exists(csv):
            print(f"SKIP (missing): {label} -> {csv}")
            continue
        df = load_run(csv)
        data[label] = (df, color, ls, ctrl)
        last = int(df["step"].max())
        print(f"loaded {label:32s} rows={len(df):>7d}  laststep={last:>9d}")

    # Figures
    fig_task(data, "fig1_task_full.png", xmax=None, tag=" (full range)")
    fig_task(data, "fig2_task_zoom50k.png", xmax=60_000, tag=" (0-60k -- controlled window)")
    fig_wm(data, "fig3_wm_losses.png")
    mlp37 = "WP 37² MLP (4887404)"
    if mlp37 in data:
        fig_health(mlp37, data[mlp37][0], "fig4_health_37mlp.png")

    # Summary table
    print("\n=== SUMMARY: rolling-mean task metrics ===")
    hdr = f"{'run':34s} {'laststep':>9s} | {'SR@50k':>7s} {'SPL@50k':>8s} {'rew@50k':>8s} | {'SR@last':>8s} {'SPL@last':>9s} {'rew@last':>9s}"
    print(hdr); print("-" * len(hdr))
    rows = []
    for label, (df, *_rest) in data.items():
        last = int(df["step"].max())
        r = dict(
            run=label, laststep=last,
            sr50=value_at(df, "metrics/sr", COMMON_STEP) * 100,
            spl50=value_at(df, "metrics/spl", COMMON_STEP),
            rew50=value_at(df, "metrics/reward", COMMON_STEP),
            srL=value_at(df, "metrics/sr", last) * 100,
            splL=value_at(df, "metrics/spl", last),
            rewL=value_at(df, "metrics/reward", last),
        )
        rows.append(r)
        print(f"{label:34s} {last:>9d} | {r['sr50']:>6.1f}% {r['spl50']:>8.3f} {r['rew50']:>8.2f} "
              f"| {r['srL']:>7.1f}% {r['splL']:>9.3f} {r['rewL']:>9.2f}")
    return rows


if __name__ == "__main__":
    main()
