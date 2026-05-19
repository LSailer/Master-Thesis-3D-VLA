"""Plotting utilities for VGGT comparison."""

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def plot_comparison(
    df: pd.DataFrame,
    output_dir: str | Path = "outputs/vggt_comparison",
) -> list[Path]:
    """Generate latency and memory plots from the comparison DataFrame.

    Returns list of saved image paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    # Latency plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for variant in df["variant"].unique():
        vdf = df[df["variant"] == variant]
        ax.plot(vdf["n_frames"], vdf["latency_ms"], marker="o", label=variant)
    ax.set_xlabel("Sequence Length (frames)")
    ax.set_ylabel("Latency (ms/frame)")
    ax.set_title("Inference Latency by Variant")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = output_dir / "latency_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    # Memory plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for variant in df["variant"].unique():
        vdf = df[df["variant"] == variant]
        ax.plot(vdf["n_frames"], vdf["peak_mem_mb"], marker="s", label=variant)
    ax.set_xlabel("Sequence Length (frames)")
    ax.set_ylabel("Peak GPU Memory (MB)")
    ax.set_title("GPU Memory Usage by Variant")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = output_dir / "memory_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)

    return saved
