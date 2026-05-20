"""Generate publication-quality plots for the random baseline experiment.

Reads output/runs/baselines/random-l1/ and saves figures to output/methods/comparisons/figures/.
"""

import json
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
    "font.size": 12,
})

ACCENT = "#6366f1"
ACCENT2 = "#818cf8"
GREEN = "#34d399"
RED = "#f87171"
AMBER = "#fbbf24"
CYAN = "#22d3ee"

INPUT_DIR = "output/runs/baselines/random-l1"
OUTPUT_DIR = "output/methods/comparisons/figures"


def plot_action_distribution(summary: dict, out_path: str):
    actions = summary["action_distribution"]
    names = list(actions.keys())
    pcts = list(actions.values())
    colors = [RED, ACCENT, CYAN, AMBER]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, pcts, color=colors, width=0.6, edgecolor="none")
    ax.axhline(25, color="#94a3b8", linestyle="--", alpha=0.5, label="Expected (25%)")
    ax.set_ylabel("Frequency (%)")
    ax.set_title("Action Distribution — Random Agent", fontweight="bold", fontsize=14)
    ax.set_ylim(0, 35)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{pct:.1f}%", ha="center", va="bottom", fontsize=11, color="#e2e8f0")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_reward_distribution(df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["reward"], bins=40, color=ACCENT, edgecolor="#1a1a24", alpha=0.85)
    ax.axvline(df["reward"].mean(), color=AMBER, linestyle="--", linewidth=2,
               label=f"Mean: {df['reward'].mean():.2f}")
    ax.set_xlabel("Episode Reward")
    ax.set_ylabel("Count")
    ax.set_title("Reward Distribution — Random Agent (834 episodes)", fontweight="bold", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_episode_length_distribution(df: pd.DataFrame, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["steps"], bins=30, color=CYAN, edgecolor="#1a1a24", alpha=0.85)
    ax.axvline(df["steps"].mean(), color=AMBER, linestyle="--", linewidth=2,
               label=f"Mean: {df['steps'].mean():.0f}")
    ax.set_xlabel("Episode Length (steps)")
    ax.set_ylabel("Count")
    ax.set_title("Episode Length Distribution — Random Agent", fontweight="bold", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_success_episodes(df: pd.DataFrame, out_path: str):
    successes = df[df["success"] > 0]
    failures = df[df["success"] == 0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(failures["steps"], failures["reward"], color="#2d2d3d", alpha=0.3,
               s=10, label=f"Failure ({len(failures)})")
    if len(successes) > 0:
        ax.scatter(successes["steps"], successes["reward"], color=GREEN, alpha=0.9,
                   s=40, edgecolors="white", linewidths=0.5,
                   label=f"Success ({len(successes)})")
    ax.set_xlabel("Episode Length (steps)")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Reward vs Steps — Success Highlighted", fontweight="bold", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(INPUT_DIR, "summary.json")) as f:
        summary = json.load(f)
    df = pd.read_csv(os.path.join(INPUT_DIR, "episodes.csv"))

    print(f"Loaded {len(df)} episodes from {INPUT_DIR}")

    plot_action_distribution(summary, os.path.join(OUTPUT_DIR, "random-l1-action-dist.png"))
    plot_reward_distribution(df, os.path.join(OUTPUT_DIR, "random-l1-reward-dist.png"))
    plot_episode_length_distribution(df, os.path.join(OUTPUT_DIR, "random-l1-episode-length.png"))
    plot_success_episodes(df, os.path.join(OUTPUT_DIR, "random-l1-success-scatter.png"))

    print(f"\nAll plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
