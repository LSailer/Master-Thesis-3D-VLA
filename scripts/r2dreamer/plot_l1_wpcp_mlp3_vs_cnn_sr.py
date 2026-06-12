"""Level-1 success-rate curve: WP+CP flatten depth-3 MLP vs references.

Thesis figure comparing the current WP+CP flatten readout with a depth-3 MLP
against the older WP+CP linear readout and the RGB CNN L1 baseline.

    MPLCONFIGDIR=/scratch/matplotlib .venv/bin/python \
      scripts/r2dreamer/plot_l1_wpcp_mlp3_vs_cnn_sr.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "../..")
OUT = os.path.join(ROOT, "../writing/img/l1-vggt-wpcp-mlp3-vs-cnn-sr.png")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#cbd5e1",
    "axes.labelcolor": "#111827",
    "text.color": "#111827",
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "grid.color": "#e2e8f0",
    "font.family": "sans-serif",
    "font.size": 12,
})

ORANGE = "#d97706"
BLUE = "#4f46e5"
GREEN = "#059669"
RED = "#dc2626"

# (label, csv, color, reported_sr). reported_sr is the final logged SR rounded
# to whole percent so the legend stays readable.
RUNS = [
    ("CNN baseline (RGB 3$\\times$64$^2$)",
     "output/runs/r2dreamer-curriculum-l1-rerun/run-3957651/metrics.csv", ORANGE, 75),
    ("WP+CP flatten, linear readout",
     "output/r2dreamer-curriculum-l1-vggt/run-4216462/metrics.csv", BLUE, 61),
    ("WP+CP flatten, 3-layer MLP",
     "output/r2dreamer-curriculum-l1-vggt-wp-cp-mlp/run-4887404/metrics.csv", GREEN, 73),
]
ROLL = 50


def sr(csv):
    df = pd.read_csv(os.path.join(ROOT, csv))
    sub = df[df["metric"] == "metrics/sr"].copy()
    sub["step"] = pd.to_numeric(sub["step"], errors="coerce")
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    sub = sub.dropna().sort_values("step")
    sub["smooth"] = sub["value"].rolling(ROLL, min_periods=1).mean()
    return sub


def main():
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, csv, color, reported in RUNS:
        s = sr(csv)
        ax.plot(s["step"] / 1e6, s["value"] * 100, color=color, alpha=0.10, linewidth=0.5)
        ax.plot(s["step"] / 1e6, s["smooth"] * 100, color=color, linewidth=2.5,
                label=f"{label} ({reported}%)")

    ax.axhline(3.84, color=RED, linestyle="--", linewidth=1.5, label="Random (3.8%)")
    ax.set_xlabel("Environment Steps (M)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("L1 ObjectNav -- WP+CP Flatten Readout Depth",
                 fontweight="bold", fontsize=14)
    ax.set_ylim(0, 100)
    ax.set_xlim(left=0)
    ax.grid(alpha=0.8)
    ax.legend(fontsize=10, loc="upper left", frameon=True, facecolor="white",
              edgecolor="#cbd5e1", framealpha=1.0)
    fig.tight_layout()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
