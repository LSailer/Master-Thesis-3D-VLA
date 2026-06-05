"""Level-1 success-rate curve: VGGT WP+CP (flatten) vs the CNN L1 baseline.

Thesis figure for content/Experiments.tex, section "3D-VLA Baseline: VGGT WP+CP
Flatten" (sec:vggt-wpcp-flatten). Matches the dark style of
plot_curriculum_scaling.py and reuses the SAME canonical runs the thesis cites:

  CNN  L1 baseline : r2d-L1-buffix-3957651  (wandb y5a0upzd) -> 75% SR, 9105 ep
  VGGT WP+CP flat  : vggt_jax-4216462        (wandb er5ze5m6) -> 62% SR, 2.137M

Success rate is metrics/sr (rolling success rate); the legend reports the final
logged value, matching the wandb run summaries the thesis prose quotes.

    python scripts/r2dreamer/plot_l1_wpcp_vs_cnn_sr.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "../..")
# Write straight into the (separate) thesis repo's image dir.
OUT = os.path.join(ROOT, "../writing/img/l1-vggt-wpcp-vs-cnn-sr.png")

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
    "font.size": 12,
})
ACCENT = "#6366f1"   # VGGT WP+CP (the variant under study)
AMBER = "#fbbf24"    # CNN baseline (reference)
RED = "#f87171"

# (label, csv, color, reported_sr). reported_sr is the wandb run-summary SR the
# thesis prose cites; the curve itself is plotted from the CSV. They differ by
# <=1 pt (summary vs last logged point) -- we show the summary so the legend
# matches Experiments.tex (CNN 75%, VGGT 62%).
RUNS = [
    ("CNN baseline (RGB 3$\\times$64$^2$)",
     "output/runs/r2dreamer-curriculum-l1-rerun/run-3957651/metrics.csv", AMBER, 75),
    ("VGGT WP+CP flatten (37$^2$+pose)",
     "output/r2dreamer-curriculum-l1-vggt/run-4216462/metrics.csv", ACCENT, 62),
]
ROLL = 50  # matches plot_curriculum_scaling.py


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
        ax.plot(s["step"] / 1e6, s["value"] * 100, color=color, alpha=0.1, linewidth=0.5)
        ax.plot(s["step"] / 1e6, s["smooth"] * 100, color=color, linewidth=2.5,
                label=f"{label} ({reported}%)")
    ax.axhline(3.84, color=RED, linestyle="--", linewidth=1.5, label="Random (3.8%)")
    ax.set_xlabel("Environment Steps (M)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("L1 ObjectNav — World Model from Geometry vs RGB",
                 fontweight="bold", fontsize=14)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_ylim(0, 100)
    ax.set_xlim(left=0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
