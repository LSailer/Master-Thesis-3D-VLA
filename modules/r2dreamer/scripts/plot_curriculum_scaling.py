"""Generate publication-quality plots for the curriculum scaling experiments.

L1-rerun (1 house, 1 goal) → L2 (1 house, 6 goals) → L3 (10 houses, 1 goal).
Includes semantic floor plan of fK2vEV32Lag for L2 goal difficulty analysis.

Reads from:
  output/r2dreamer-curriculum-l1-rerun/run-3957651/metrics.csv
  output/r2dreamer-curriculum-l2/run-3957713/metrics.csv
  output/r2dreamer-curriculum-l3/run-3957714/metrics.csv
  output/floorplan_fK2vEV32Lag.pkl
Saves to: output/figures/
"""

import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
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
    "font.size": 12,
})

ACCENT = "#6366f1"
ACCENT2 = "#818cf8"
GREEN = "#34d399"
RED = "#f87171"
AMBER = "#fbbf24"
CYAN = "#22d3ee"
PINK = "#f472b6"
ORANGE = "#fb923c"

# Goal category colors (consistent across all plots)
GOAL_COLORS = {
    "plant": GREEN,
    "bed": ACCENT2,
    "chair": CYAN,
    "sofa": AMBER,
    "toilet": ORANGE,
    "tv_monitor": RED,
}

# Semantic category name mapping (HM3D names → ObjectNav names)
SEMANTIC_TO_GOAL = {
    "bed": "bed",
    "chair": "chair",
    "plant": "plant",
    "couch": "sofa",
    "toilet": "toilet",
    "tv": "tv_monitor",
}

L1_CSV = "output/r2dreamer-curriculum-l1-rerun/run-3957651/metrics.csv"
L2_CSV = "output/r2dreamer-curriculum-l2/run-3957713/metrics.csv"
L3_CSV = "output/r2dreamer-curriculum-l3/run-3957714/metrics.csv"
FLOORPLAN_PKL = "output/floorplan_fK2vEV32Lag.pkl"
OUTPUT_DIR = "output/figures"


def load_metric(df, name, rolling=0):
    sub = df[df["metric"] == name].sort_values("step").copy()
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    sub = sub.dropna(subset=["value"])
    if rolling > 0 and len(sub) > rolling:
        sub["smooth"] = sub["value"].rolling(rolling, min_periods=1).mean()
    else:
        sub["smooth"] = sub["value"]
    return sub


def plot_sr_comparison(df_l1, df_l2, df_l3, out_path):
    """SR training curves for all three experiments."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for df, label, color in [
        (df_l1, "L1: 1 house, chair (75%)", ACCENT),
        (df_l2, "L2: 1 house, 6 goals (36%)", AMBER),
        (df_l3, "L3: 10 houses, chair (32%)", GREEN),
    ]:
        sr = load_metric(df, "metrics/sr", rolling=50)
        sr["pct"] = sr["smooth"] * 100
        ax.plot(sr["step"] / 1e6, sr["value"] * 100, color=color, alpha=0.1, linewidth=0.5)
        ax.plot(sr["step"] / 1e6, sr["pct"], color=color, linewidth=2.5, label=label)

    ax.axhline(3.84, color=RED, linestyle="--", linewidth=1.5, label="Random (3.8%)")
    ax.set_xlabel("Environment Steps (M)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate — Curriculum Scaling", fontweight="bold", fontsize=14)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_spl_comparison(df_l1, df_l2, df_l3, out_path):
    """SPL training curves for all three experiments."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for df, label, color in [
        (df_l1, "L1: 1 house, chair (0.55)", ACCENT),
        (df_l2, "L2: 1 house, 6 goals (0.25)", AMBER),
        (df_l3, "L3: 10 houses, chair (0.21)", GREEN),
    ]:
        spl = load_metric(df, "metrics/spl", rolling=50)
        ax.plot(spl["step"] / 1e6, spl["value"], color=color, alpha=0.1, linewidth=0.5)
        ax.plot(spl["step"] / 1e6, spl["smooth"], color=color, linewidth=2.5, label=label)

    ax.axhline(0.023, color=RED, linestyle="--", linewidth=1.5, label="Random (0.023)")
    ax.set_xlabel("Environment Steps (M)")
    ax.set_ylabel("SPL")
    ax.set_title("SPL — Curriculum Scaling", fontweight="bold", fontsize=14)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_l2_per_goal_sr(df_l2, out_path):
    """Per-goal SR curves for L2 showing the difficulty hierarchy."""
    fig, ax = plt.subplots(figsize=(10, 5))

    goals = ["plant", "bed", "chair", "sofa", "toilet", "tv_monitor"]
    for goal in goals:
        sr = load_metric(df_l2, f"goal/{goal}/sr", rolling=30)
        if len(sr) == 0:
            continue
        sr["pct"] = sr["smooth"] * 100
        ax.plot(sr["step"] / 1e6, sr["value"] * 100, color=GOAL_COLORS[goal],
                alpha=0.1, linewidth=0.5)
        ax.plot(sr["step"] / 1e6, sr["pct"], color=GOAL_COLORS[goal],
                linewidth=2.5, label=f"{goal}")

    ax.axhline(3.84, color="#94a3b8", linestyle="--", linewidth=1, alpha=0.5,
               label="Random")
    ax.set_xlabel("Environment Steps (M)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("L2 Per-Goal Success Rate — Goal Difficulty Hierarchy",
                 fontweight="bold", fontsize=14)
    ax.legend(fontsize=10, ncol=2, loc="upper left")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_l2_goal_bar(out_path):
    """Bar chart: per-goal SR with Geo/Euc ratio annotation."""
    goals = ["plant", "bed", "chair", "sofa", "toilet", "tv_monitor"]
    srs = [66, 59, 46, 38, 11, 3]
    ratios = [1.18, 1.11, 1.15, 1.20, 1.29, 1.77]
    colors = [GOAL_COLORS[g] for g in goals]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(goals, srs, color=colors, edgecolor="#2d2d3d", linewidth=1.5)

    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"Geo/Euc\n{ratio:.2f}", ha="center", va="bottom",
                fontsize=9, color="#94a3b8")

    ax.axhline(3.84, color=RED, linestyle="--", linewidth=1.5, alpha=0.7,
               label="Random (3.8%)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("L2 Goal Difficulty — Navigation Complexity Predicts SR",
                 fontweight="bold", fontsize=14)
    ax.set_ylim(0, 85)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_semantic_floorplan(out_path):
    """Semantic floor plan of fK2vEV32Lag with goal objects and start positions."""
    with open(FLOORPLAN_PKL, "rb") as f:
        data = pickle.load(f)

    nav = data["navmesh"]
    grid = nav["grid"]
    sem_grid = data["semantic_grid"]
    id_to_cat = data["id_to_cat"]
    x_min, x_max = nav["x_min"], nav["x_max"]
    z_min, z_max = nav["z_min"], nav["z_max"]
    res = nav["resolution"]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Layer 1: Navigable area (dark gray)
    nav_rgba = np.zeros((*grid.shape, 4))
    nav_rgba[grid, :] = [0.15, 0.15, 0.2, 1.0]  # navigable = dark
    nav_rgba[~grid, :] = [0.06, 0.06, 0.08, 1.0]  # walls = darker
    ax.imshow(nav_rgba, extent=[x_min, x_max, z_max, z_min],
              aspect="equal", interpolation="nearest")

    # Layer 2: Semantic objects (semi-transparent colored regions)
    goal_cats = set(SEMANTIC_TO_GOAL.keys())
    sem_overlay = np.zeros((*sem_grid.shape, 4))
    for int_id, cat_name in id_to_cat.items():
        if cat_name not in goal_cats:
            continue
        goal_name = SEMANTIC_TO_GOAL[cat_name]
        color = GOAL_COLORS[goal_name]
        # Convert hex to RGB
        r = int(color[1:3], 16) / 255
        g = int(color[3:5], 16) / 255
        b = int(color[5:7], 16) / 255
        mask = sem_grid == int_id
        sem_overlay[mask] = [r, g, b, 0.6]

    ax.imshow(sem_overlay, extent=[x_min, x_max, z_max, z_min],
              aspect="equal", interpolation="nearest")

    # Layer 3: Goal object positions (large markers)
    goal_final_sr = {
        "plant": 66, "bed": 59, "chair": 46,
        "sofa": 38, "toilet": 11, "tv_monitor": 3,
    }
    for cat, positions in data["goals_by_cat"].items():
        for pos in positions:
            x, z = pos[0], pos[2]
            ax.plot(x, z, marker="*", markersize=18, color=GOAL_COLORS[cat],
                    markeredgecolor="white", markeredgewidth=1.5, zorder=10)
            sr = goal_final_sr.get(cat, "?")
            ax.annotate(f"{cat}\n{sr}% SR", (x, z), fontsize=8,
                        fontweight="bold", color="white",
                        ha="center", va="bottom",
                        xytext=(0, 12), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  facecolor=GOAL_COLORS[cat], alpha=0.8,
                                  edgecolor="none"),
                        zorder=11)

    # Layer 4: Start position density (small dots, sampled for clarity)
    rng = np.random.RandomState(42)
    for cat, starts in data["starts_by_cat"].items():
        starts = np.array(starts)
        # Sample max 200 for visual clarity
        if len(starts) > 200:
            idx = rng.choice(len(starts), 200, replace=False)
            starts = starts[idx]
        ax.scatter(starts[:, 0], starts[:, 2], s=3, alpha=0.15,
                   color=GOAL_COLORS[cat], zorder=5)

    # Legend
    legend_elements = [
        Patch(facecolor="#262632", edgecolor="#94a3b8", label="Navigable area"),
    ]
    for goal in ["plant", "bed", "chair", "sofa", "toilet", "tv_monitor"]:
        legend_elements.append(
            Line2D([0], [0], marker="*", color="w", markerfacecolor=GOAL_COLORS[goal],
                   markersize=12, label=f"{goal} ({goal_final_sr[goal]}% SR)",
                   linestyle="None")
        )
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9,
              facecolor="#1a1a24", edgecolor="#2d2d3d", framealpha=0.95)

    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Z (meters)")
    ax.set_title("Semantic Floor Plan — fK2vEV32Lag (L2 Scene)\n"
                 "Object accessibility explains goal difficulty hierarchy",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_wm_losses_comparison(df_l1, df_l2, df_l3, out_path):
    """World model dynamics loss comparison (train vs val) across experiments."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, df, label, color in [
        (axes[0], df_l1, "L1 Rerun", ACCENT),
        (axes[1], df_l2, "L2", AMBER),
        (axes[2], df_l3, "L3", GREEN),
    ]:
        dyn = load_metric(df, "loss/dyn", rolling=50)
        val_dyn = load_metric(df, "val/loss/dyn")
        ax.plot(dyn["step"] / 1e6, dyn["smooth"], color=color, linewidth=2,
                label="Train")
        if len(val_dyn) > 0:
            ax.plot(val_dyn["step"] / 1e6, val_dyn["value"], color=RED,
                    linewidth=2, marker="o", markersize=3, label="Val")
        ax.set_xlabel("Steps (M)")
        ax.set_title(label, fontweight="bold", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Dynamics Loss (KL)")
    fig.suptitle("World Model Dynamics Loss — Train vs Val",
                 fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading metrics...")
    df_l1 = pd.read_csv(L1_CSV)
    df_l2 = pd.read_csv(L2_CSV)
    df_l3 = pd.read_csv(L3_CSV)
    print(f"  L1: {len(df_l1)} rows, L2: {len(df_l2)} rows, L3: {len(df_l3)} rows")

    plot_sr_comparison(df_l1, df_l2, df_l3,
                       os.path.join(OUTPUT_DIR, "curriculum-sr-comparison.png"))
    plot_spl_comparison(df_l1, df_l2, df_l3,
                        os.path.join(OUTPUT_DIR, "curriculum-spl-comparison.png"))
    plot_l2_per_goal_sr(df_l2,
                        os.path.join(OUTPUT_DIR, "l2-per-goal-sr.png"))
    plot_l2_goal_bar(os.path.join(OUTPUT_DIR, "l2-goal-bar.png"))
    plot_semantic_floorplan(os.path.join(OUTPUT_DIR, "l2-semantic-floorplan.png"))
    plot_wm_losses_comparison(df_l1, df_l2, df_l3,
                              os.path.join(OUTPUT_DIR, "curriculum-wm-losses.png"))

    print(f"\nAll plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
