"""Plot training curves and diagnostics for R2-Dreamer baseline run.

Reads metrics.csv from the baseline run output directory and generates
publication-quality figures for thesis slides and wiki pages.

Usage:
    uv run python scripts/r2dreamer/plot_baseline_analysis.py
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = Path("output/runs/r2dreamer-habitat-baseline/run-3907457/metrics.csv")
OUT_DIR = Path("output/methods/comparisons/figures")

# Thesis-quality defaults
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.figsize": (10, 6),
})


def load_metrics(path: str) -> dict[str, list[tuple[int, float]]]:
    """Load CSV into {metric_name: [(step, value), ...]}."""
    metrics = defaultdict(list)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step"])
            val = float(row["value"])
            if val == val:  # skip NaN
                metrics[row["metric"]].append((step, val))
    return metrics


def rolling_mean(steps, values, window=50):
    """Compute rolling mean for noisy time series."""
    if len(values) < window:
        return np.array(steps), np.array(values)
    cumsum = np.cumsum(np.insert(values, 0, 0))
    smoothed = (cumsum[window:] - cumsum[:-window]) / window
    return np.array(steps[window - 1:]), smoothed


def _plot_episode_reward(ax, metrics: dict):
    steps, vals = zip(*metrics["episode/reward"])
    ax.scatter(steps, vals, alpha=0.1, s=2, c="steelblue")
    s_steps, s_vals = rolling_mean(list(steps), list(vals), window=100)
    ax.plot(s_steps, s_vals, color="darkblue", linewidth=2, label="100-ep avg")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Episode Reward (geodesic delta)")
    ax.legend()


def _plot_success_rate(ax, metrics: dict):
    steps, vals = zip(*metrics["episode/success"])
    s_steps, s_vals = rolling_mean(list(steps), list(vals), window=100)
    ax.plot(s_steps, s_vals * 100, color="green", linewidth=2)
    ax.axhline(y=2.36, color="red", linestyle="--", alpha=0.5, label="Overall avg (2.36%)")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate (100-episode rolling)")
    ax.set_ylim(-0.5, 15)
    ax.legend()


def _plot_episode_steps(ax, metrics: dict):
    steps, vals = zip(*metrics["episode/steps"])
    ax.scatter(steps, vals, alpha=0.1, s=2, c="coral")
    s_steps, s_vals = rolling_mean(list(steps), list(vals), window=100)
    ax.plot(s_steps, s_vals, color="darkred", linewidth=2, label="100-ep avg")
    ax.axhline(y=500, color="gray", linestyle="--", alpha=0.3, label="Max (500)")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Steps per Episode")
    ax.set_title("Episode Length")
    ax.legend()


def _success_quartiles(metrics: dict) -> list[float]:
    success_data = list(zip(*metrics["episode/success"]))
    n = len(success_data[1])
    q_size = n // 4
    quartiles = []
    for i in range(4):
        start = i * q_size
        end = (i + 1) * q_size if i < 3 else n
        chunk = success_data[1][start:end]
        sr = sum(chunk) / len(chunk) * 100
        quartiles.append(sr)
    return quartiles


def _plot_success_quartiles(ax, metrics: dict):
    quartiles = _success_quartiles(metrics)
    bars = ax.bar(["0-600K", "600K-1.2M", "1.2M-1.8M", "1.8M-2.4M"],
                  quartiles, color=["#4c72b0", "#55a868", "#c44e52", "#8172b2"])
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate by Training Quartile")
    for bar, val in zip(bars, quartiles):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10)


def plot_episode_metrics(metrics: dict, out_dir: Path):
    """Plot reward, success rate, and episode length over training."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    _plot_episode_reward(axes[0, 0], metrics)
    _plot_success_rate(axes[0, 1], metrics)
    _plot_episode_steps(axes[1, 0], metrics)
    _plot_success_quartiles(axes[1, 1], metrics)

    fig.suptitle("R2-Dreamer Baseline — 2.4M Steps, All Scenes, No Goal Conditioning",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "baseline_episode_metrics.png", bbox_inches="tight")
    plt.close()
    print(f"Saved {out_dir / 'baseline_episode_metrics.png'}")


def plot_world_model_losses(metrics: dict, out_dir: Path):
    """Plot world model training losses over time."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    loss_configs = [
        ("loss/dyn", "KL / Dynamics Loss", "tab:blue", axes[0, 0]),
        ("loss/rew", "Reward Prediction Loss", "tab:orange", axes[0, 1]),
        ("total_loss", "Total Loss", "tab:red", axes[1, 0]),
        ("loss/rep", "Representation KL Loss", "tab:purple", axes[1, 1]),
    ]

    for metric_name, title, color, ax in loss_configs:
        if metric_name in metrics:
            steps, vals = zip(*metrics[metric_name])
            ax.plot(steps, vals, alpha=0.3, linewidth=0.5, color=color)
            s_steps, s_vals = rolling_mean(list(steps), list(vals), window=20)
            ax.plot(s_steps, s_vals, color=color, linewidth=2, label="Smoothed")
            ax.set_xlabel("Environment Steps")
            ax.set_ylabel("Loss")
            ax.set_title(title)
            ax.legend()

    fig.suptitle("R2-Dreamer Baseline — World Model Losses",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "baseline_wm_losses.png", bbox_inches="tight")
    plt.close()
    print(f"Saved {out_dir / 'baseline_wm_losses.png'}")


def plot_policy_diagnostics(metrics: dict, out_dir: Path):
    """Plot policy loss and entropy over time."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Policy loss
    ax = axes[0]
    if "loss/policy" in metrics:
        steps, vals = zip(*metrics["loss/policy"])
        ax.plot(steps, vals, alpha=0.3, linewidth=0.5, color="teal")
        s_steps, s_vals = rolling_mean(list(steps), list(vals), window=20)
        ax.plot(s_steps, s_vals, color="teal", linewidth=2)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Policy Loss")
    ax.set_title("Actor Loss (flat = no policy improvement)")

    # Latent entropy
    ax = axes[1]
    for key, label, color in [
        ("latent/prior_entropy", "Prior Entropy", "tab:blue"),
        ("latent/posterior_entropy", "Posterior Entropy", "tab:orange"),
    ]:
        if key in metrics:
            steps, vals = zip(*metrics[key])
            s_steps, s_vals = rolling_mean(list(steps), list(vals), window=20)
            ax.plot(s_steps, s_vals, color=color, linewidth=2, label=label)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Entropy (nats)")
    ax.set_title("Latent State Entropy")
    ax.legend()

    fig.suptitle("R2-Dreamer Baseline — Policy & Latent Diagnostics",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "baseline_policy_diagnostics.png", bbox_inches="tight")
    plt.close()
    print(f"Saved {out_dir / 'baseline_policy_diagnostics.png'}")


def main():
    metrics = load_metrics(CSV_PATH)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {sum(len(v) for v in metrics.values())} data points "
          f"across {len(metrics)} metrics")

    plot_episode_metrics(metrics, OUT_DIR)
    plot_world_model_losses(metrics, OUT_DIR)
    plot_policy_diagnostics(metrics, OUT_DIR)

    print("\nAll plots saved to output/methods/comparisons/figures/")


if __name__ == "__main__":
    main()
