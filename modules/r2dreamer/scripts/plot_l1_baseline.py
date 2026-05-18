"""Generate publication-quality plots for the L1 curriculum baseline experiment.

Reads output/runs/r2dreamer-curriculum-l1/run-3923812/metrics.csv and
output/runs/baselines/random-l1/summary.json for comparison.
Saves figures to output/methods/comparisons/figures/.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

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

METRICS_CSV = "output/runs/r2dreamer-curriculum-l1/run-3923812/metrics.csv"
RANDOM_JSON = "output/runs/baselines/random-l1/summary.json"
OUTPUT_DIR = "output/methods/comparisons/figures"


def load_metric(df: pd.DataFrame, name: str, rolling: int = 0) -> pd.DataFrame:
    sub = df[df["metric"] == name].sort_values("step").copy()
    sub["value"] = pd.to_numeric(sub["value"], errors="coerce")
    sub = sub.dropna(subset=["value"])
    if rolling > 0 and len(sub) > rolling:
        sub["smooth"] = sub["value"].rolling(rolling, min_periods=1).mean()
    else:
        sub["smooth"] = sub["value"]
    return sub


def plot_sr_vs_random(df: pd.DataFrame, random_sr: float, out_path: str):
    sr = load_metric(df, "metrics/sr", rolling=50)
    sr["value_pct"] = sr["value"] * 100
    sr["smooth_pct"] = sr["smooth"] * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sr["step"] / 1e6, sr["value_pct"], color=ACCENT, alpha=0.2, linewidth=0.5)
    ax.plot(sr["step"] / 1e6, sr["smooth_pct"], color=ACCENT, linewidth=2, label="R2-Dreamer L1 (rolling 50)")
    ax.axhline(random_sr * 100, color=RED, linestyle="--", linewidth=2, label=f"Random baseline ({random_sr*100:.1f}%)")
    ax.set_xlabel("Environment Steps (M)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate — R2-Dreamer L1 vs Random Baseline", fontweight="bold", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_spl(df: pd.DataFrame, random_spl: float, out_path: str):
    spl = load_metric(df, "metrics/spl", rolling=50)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(spl["step"] / 1e6, spl["value"], color=CYAN, alpha=0.2, linewidth=0.5)
    ax.plot(spl["step"] / 1e6, spl["smooth"], color=CYAN, linewidth=2, label="R2-Dreamer L1 (rolling 50)")
    ax.axhline(random_spl, color=RED, linestyle="--", linewidth=2, label=f"Random baseline ({random_spl:.3f})")
    ax.set_xlabel("Environment Steps (M)")
    ax.set_ylabel("SPL")
    ax.set_title("SPL — R2-Dreamer L1 vs Random Baseline", fontweight="bold", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_world_model_losses(df: pd.DataFrame, out_path: str):
    dyn = load_metric(df, "loss/dyn", rolling=50)
    rew = load_metric(df, "loss/rew", rolling=50)
    val_dyn = load_metric(df, "val/loss/dyn")
    val_rew = load_metric(df, "val/loss/rew")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Dynamics loss
    ax = axes[0]
    ax.plot(dyn["step"] / 1e6, dyn["smooth"], color=ACCENT, linewidth=2, label="Train dyn (KL)")
    ax.plot(val_dyn["step"] / 1e6, val_dyn["value"], color=RED, linewidth=2, marker="o", markersize=3, label="Val dyn (KL)")
    ax.set_xlabel("Environment Steps (M)")
    ax.set_ylabel("Loss")
    ax.set_title("Dynamics Loss (KL Divergence)", fontweight="bold", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Reward loss
    ax = axes[1]
    ax.plot(rew["step"] / 1e6, rew["smooth"], color=GREEN, linewidth=2, label="Train reward")
    ax.plot(val_rew["step"] / 1e6, val_rew["value"], color=AMBER, linewidth=2, marker="o", markersize=3, label="Val reward")
    ax.set_xlabel("Environment Steps (M)")
    ax.set_ylabel("Loss")
    ax.set_title("Reward Prediction Loss", fontweight="bold", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_policy_and_kl(df: pd.DataFrame, out_path: str):
    policy = load_metric(df, "loss/policy", rolling=50)
    value = load_metric(df, "loss/value", rolling=50)
    kl = load_metric(df, "loss/dyn", rolling=50)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Policy + value loss
    ax = axes[0]
    ax.plot(policy["step"] / 1e6, policy["smooth"], color=ACCENT, linewidth=2, label="Policy loss")
    ax.plot(value["step"] / 1e6, value["smooth"], color=GREEN, linewidth=2, label="Value loss")
    ax.axhline(0, color="#94a3b8", linestyle=":", alpha=0.5)
    ax.set_xlabel("Environment Steps (M)")
    ax.set_ylabel("Loss")
    ax.set_title("Actor-Critic Losses", fontweight="bold", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # KL divergence
    ax = axes[1]
    ax.plot(kl["step"] / 1e6, kl["smooth"], color=AMBER, linewidth=2, label="Dynamics KL")
    ax.set_xlabel("Environment Steps (M)")
    ax.set_ylabel("KL (nats)")
    ax.set_title("Latent KL Divergence", fontweight="bold", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_episode_metrics(df: pd.DataFrame, out_path: str):
    reward = load_metric(df, "episode/reward", rolling=50)
    steps = load_metric(df, "episode/steps", rolling=50)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(reward["step"] / 1e6, reward["value"], color=ACCENT, alpha=0.1, linewidth=0.5)
    ax.plot(reward["step"] / 1e6, reward["smooth"], color=ACCENT, linewidth=2, label="Episode reward (rolling 50)")
    ax.axhline(0, color="#94a3b8", linestyle=":", alpha=0.5)
    ax.set_xlabel("Environment Steps (M)")
    ax.set_ylabel("Reward")
    ax.set_title("Episode Reward", fontweight="bold", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(steps["step"] / 1e6, steps["value"], color=CYAN, alpha=0.1, linewidth=0.5)
    ax.plot(steps["step"] / 1e6, steps["smooth"], color=CYAN, linewidth=2, label="Episode steps (rolling 50)")
    ax.set_xlabel("Environment Steps (M)")
    ax.set_ylabel("Steps")
    ax.set_title("Episode Length", fontweight="bold", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(METRICS_CSV)
    with open(RANDOM_JSON) as f:
        random_summary = json.load(f)

    print(f"Loaded {len(df)} metric rows from {METRICS_CSV}")

    plot_sr_vs_random(df, random_summary["sr"], os.path.join(OUTPUT_DIR, "l1-baseline-sr.png"))
    plot_spl(df, random_summary["spl"], os.path.join(OUTPUT_DIR, "l1-baseline-spl.png"))
    plot_world_model_losses(df, os.path.join(OUTPUT_DIR, "l1-baseline-wm-losses.png"))
    plot_policy_and_kl(df, os.path.join(OUTPUT_DIR, "l1-baseline-policy-kl.png"))
    plot_episode_metrics(df, os.path.join(OUTPUT_DIR, "l1-baseline-episodes.png"))

    print(f"\nAll plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
