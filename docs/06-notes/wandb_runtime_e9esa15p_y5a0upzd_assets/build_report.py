#!/usr/bin/env python3
"""Build an HTML runtime comparison report for two W&B runs."""

from __future__ import annotations

import json
import math
import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb


PROJECT = "sailer-luca-university-ulm/3d-vla-objectnav"
RUN_IDS = ["e9esa15p", "y5a0upzd"]
RUN_LABELS = {
    "e9esa15p": "e9esa15p - timeout run",
    "y5a0upzd": "y5a0upzd - completed baseline",
}
ROOT = Path(__file__).resolve().parents[3]
ASSET_DIR = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "docs/notes/wandb_runtime_e9esa15p_y5a0upzd.html"
LOCAL_PATHS = {
    "e9esa15p": {
        "manifest": ROOT
        / "output/r2dreamer-l1-replay-capacity/cnn-cap1m-seed4367942/run-4974622/MANIFEST.json",
        "metrics": ROOT
        / "output/r2dreamer-l1-replay-capacity/cnn-cap1m-seed4367942/run-4974622/metrics.csv",
        "slurm_out": ROOT
        / "output/r2dreamer-l1-replay-capacity/cnn-cap1m-seed4367942/slurm-4974622.out",
        "slurm_err": ROOT
        / "output/r2dreamer-l1-replay-capacity/cnn-cap1m-seed4367942/slurm-4974622.err",
    },
    "y5a0upzd": {
        "metrics": ROOT / "output/runs/r2dreamer-curriculum-l1-rerun/run-3957651/metrics.csv",
        "slurm_out": ROOT / "output/runs/r2dreamer-curriculum-l1-rerun/slurm-3957651.out",
        "slurm_err": ROOT / "output/runs/r2dreamer-curriculum-l1-rerun/slurm-3957651.err",
    },
}

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
    "blue": "#5477C4",
    "orange": "#CC6F47",
    "gold": "#B8A037",
}


@dataclass
class RunBundle:
    run_id: str
    run: Any
    history: pd.DataFrame
    episodes: pd.DataFrame
    system: pd.DataFrame
    metadata: dict[str, Any]
    manifest: dict[str, Any]
    slurm: dict[str, Any]


def scan(run: Any, keys: list[str]) -> pd.DataFrame:
    rows = list(run.scan_history(keys=keys, page_size=5000))
    if not rows:
        return pd.DataFrame(columns=keys)
    return pd.DataFrame(rows)


def scan_metric_series(run: Any, metric_key: str) -> pd.DataFrame:
    df = scan(run, ["_runtime", metric_key])
    if df.empty or metric_key not in df:
        return pd.DataFrame(columns=["_runtime", metric_key])
    df = df[["_runtime", metric_key]].dropna(subset=[metric_key]).copy()
    df["_runtime"] = pd.to_numeric(df["_runtime"], errors="coerce")
    df[metric_key] = pd.to_numeric(df[metric_key], errors="coerce")
    return df.dropna(subset=["_runtime", metric_key])


def scan_system_metrics(run: Any) -> pd.DataFrame:
    try:
        return run.history(stream="events", samples=20000)
    except Exception:
        return pd.DataFrame(columns=["_runtime"])


def history_keys(run: Any) -> set[str]:
    keys = run._attrs.get("historyKeys") or {}
    if isinstance(keys, dict) and "keys" in keys:
        return set(keys["keys"])
    if isinstance(keys, dict):
        return set(keys)
    return set(keys or [])


def load_wandb_metadata(run: Any, run_id: str) -> dict[str, Any]:
    target = ASSET_DIR / f"{run_id}_wandb_metadata"
    target.mkdir(parents=True, exist_ok=True)
    try:
        file_obj = run.file("wandb-metadata.json")
        file_obj.download(root=str(target), replace=True)
        return json.loads((target / "wandb-metadata.json").read_text())
    except Exception as exc:  # W&B file availability varies across runs.
        return {"download_error": repr(exc)}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def parse_slurm(run_id: str) -> dict[str, Any]:
    paths = LOCAL_PATHS[run_id]
    out = paths.get("slurm_out")
    err = paths.get("slurm_err")
    text = ""
    if out and out.exists():
        text += out.read_text(errors="replace") + "\n"
    if err and err.exists():
        text += err.read_text(errors="replace") + "\n"

    fields: dict[str, Any] = {}
    patterns = {
        "job_header": r"Job (\d+) on ([\w.\-]+) at (.+)",
        "gpu": r"GPU: (.+)",
        "renderer": r"Renderer: (.+)",
        "driver": r"OpenGL version: (.+)",
        "curriculum": r"Curriculum \[(.+?)\] train: ([^\n]+)",
        "val_replay": r"ValReplayDataset: ([^\n]+)",
        "training_for": r"Training for (\d+) steps",
        "state": r"State: ([^\n]+)",
        "wall": r"Job Wall-clock time: ([^\n]+)",
        "memory": r"Memory Utilized: ([^\n]+)",
        "cpu": r"CPU Utilized: ([^\n]+)",
        "timeout": r"DUE TO TIME LIMIT",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            fields[key] = match.groups() if len(match.groups()) > 1 else match.group(1) if match.groups() else True

    fps_matches = [
        (int(step), int(total), int(fps))
        for step, total, fps in re.findall(r"\[step\s+(\d+)/(\d+)\].*?fps=(\d+)", text)
    ]
    fields["fps_rows"] = fps_matches
    if fps_matches:
        last_200 = fps_matches[-200:]
        fields["fps_last"] = fps_matches[-1][2]
        fields["fps_median_last_200"] = statistics.median([x[2] for x in last_200])
        fields["fps_max"] = max(x[2] for x in fps_matches)
        fields["fps_row_count"] = len(fps_matches)

    val_steps = [int(x) for x in re.findall(r"\[step\s+(\d+)\] VAL:", text)]
    fields["val_count"] = len(val_steps)
    fields["last_val_step"] = max(val_steps) if val_steps else None
    fields["checkpoint_count"] = len(re.findall(r"Checkpoint saved:", text))
    return fields


def dedupe_step_runtime(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.dropna(subset=["_step", "_runtime"]).copy()
    df["_step"] = df["_step"].astype(float)
    df["_runtime"] = df["_runtime"].astype(float)
    if "_timestamp" in df:
        df["_timestamp"] = df["_timestamp"].astype(float)
    return df.sort_values(["_step", "_runtime"]).groupby("_step", as_index=False).last()


def runtime_at_step(df: pd.DataFrame, step: float) -> float:
    if df.empty:
        return math.nan
    steps = df["_step"].to_numpy(dtype=float)
    runtimes = df["_runtime"].to_numpy(dtype=float)
    if step <= steps.min():
        return float(runtimes[0])
    if step >= steps.max():
        return float(runtimes[-1])
    return float(np.interp(step, steps, runtimes))


def summarize_bundle(bundle: RunBundle) -> dict[str, Any]:
    df = bundle.history
    ep = bundle.episodes
    summary = dict(bundle.run.summary)
    last_step = float(df["_step"].max()) if not df.empty else float(summary.get("_step", math.nan))
    last_runtime = float(df["_runtime"].max()) if not df.empty else float(summary.get("_runtime", math.nan))
    target = bundle.run.config.get("total_steps") or (
        bundle.slurm.get("training_for") if bundle.slurm.get("training_for") else None
    )
    target = int(target) if target is not None else None
    throughput = last_step / last_runtime if last_runtime else math.nan

    out = {
        "run_id": bundle.run_id,
        "name": bundle.run.name,
        "state": bundle.run.state,
        "url": bundle.run.url,
        "created_at": bundle.run._attrs.get("createdAt"),
        "heartbeat_at": bundle.run._attrs.get("heartbeatAt"),
        "commit": bundle.run._attrs.get("commit"),
        "last_step": last_step,
        "target_steps": target,
        "runtime_h": last_runtime / 3600 if last_runtime else math.nan,
        "steps_per_s": throughput,
        "steps_per_h": throughput * 3600 if throughput else math.nan,
        "episode_count": float(summary.get("episode/count", math.nan)),
        "episode_sr": float(summary.get("metrics/sr", summary.get("goal/chair/sr", math.nan))),
        "episode_spl": float(summary.get("metrics/spl", summary.get("goal/chair/spl", math.nan))),
        "slurm_state": bundle.slurm.get("state"),
        "slurm_wall": bundle.slurm.get("wall"),
        "gpu": bundle.slurm.get("gpu"),
        "node": bundle.slurm.get("job_header", [None, None, None])[1]
        if isinstance(bundle.slurm.get("job_header"), tuple)
        else None,
        "git_sha": bundle.manifest.get("git_sha") or bundle.run._attrs.get("commit"),
        "git_branch": bundle.manifest.get("git_branch"),
        "git_dirty": bundle.manifest.get("git_dirty"),
        "hostname": bundle.manifest.get("hostname") or bundle.metadata.get("host"),
        "python": bundle.manifest.get("python_version") or bundle.metadata.get("python"),
        "wandb_version": bundle.metadata.get("executable") or bundle.metadata.get("codePath"),
        "training_for_log": bundle.slurm.get("training_for"),
        "val_count": bundle.slurm.get("val_count"),
        "checkpoint_count": bundle.slurm.get("checkpoint_count"),
    }
    if target and throughput:
        out["projected_runtime_h_for_target"] = target / throughput / 3600
        out["projected_finish_margin_h_vs_24h"] = 24 - out["projected_runtime_h_for_target"]
        out["completion_pct"] = last_step / target * 100
    if not ep.empty:
        out["episodes_per_h"] = out["episode_count"] / out["runtime_h"] if out["runtime_h"] else math.nan
        out["mean_logged_episode_steps"] = float(ep["episode/steps"].dropna().mean())
        out["median_logged_episode_steps"] = float(ep["episode/steps"].dropna().median())
    return out


def summarize_system(df: pd.DataFrame, run_id: str) -> dict[str, Any]:
    metrics = {}
    for key in [
        "system.gpu.0.gpu",
        "system.gpu.0.memory",
        "system.gpu.0.memoryAllocatedBytes",
        "system.gpu.0.powerWatts",
        "system.gpu.0.powerPercent",
        "system.gpu.0.temp",
        "system.cpu",
        "system.proc.memory.rssMB",
        "system.memory_percent",
    ]:
        if key not in df.columns:
            continue
        series = pd.to_numeric(df[key], errors="coerce").dropna()
        if series.empty:
            continue
        value = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "p10": float(series.quantile(0.10)),
            "p90": float(series.quantile(0.90)),
            "max": float(series.max()),
            "count": int(series.count()),
        }
        if key.endswith("memoryAllocatedBytes"):
            for stat in ["mean", "median", "p10", "p90", "max"]:
                value[stat] = value[stat] / (1024**3)
        metrics[key] = value
    metrics["run_id"] = run_id
    return metrics


def add_chart_header(fig, ax, title: str, subtitle: str) -> None:
    fig.subplots_adjust(top=0.84)
    left = ax.get_position().x0
    fig.text(left, 0.96, title, ha="left", va="top", fontsize=15, fontweight="bold", color=TOKENS["ink"])
    fig.text(left, 0.91, subtitle, ha="left", va="top", fontsize=10, color=TOKENS["muted"])


def style_ax(ax) -> None:
    ax.set_facecolor(TOKENS["panel"])
    ax.grid(True, axis="y", color=TOKENS["grid"], linewidth=0.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    ax.tick_params(colors=TOKENS["muted"], labelsize=9)
    ax.xaxis.label.set_color(TOKENS["ink"])
    ax.yaxis.label.set_color(TOKENS["ink"])


def build_charts(bundles: dict[str, RunBundle], summaries: pd.DataFrame) -> dict[str, str]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.facecolor": TOKENS["surface"],
        }
    )
    chart_paths: dict[str, str] = {}

    # Trend line: cumulative training progress over runtime.
    trend_frames = []
    for run_id, bundle in bundles.items():
        df = bundle.history[["_runtime", "_step"]].copy()
        df["runtime_h"] = df["_runtime"] / 3600
        df["steps_m"] = df["_step"] / 1_000_000
        df["run"] = RUN_LABELS[run_id]
        trend_frames.append(df)
    trend = pd.concat(trend_frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=160, facecolor=TOKENS["surface"])
    palette = {
        RUN_LABELS["e9esa15p"]: TOKENS["orange"],
        RUN_LABELS["y5a0upzd"]: TOKENS["blue"],
    }
    for label, group in trend.groupby("run"):
        ax.plot(group["runtime_h"], group["steps_m"], color=palette[label], linewidth=2, label=label)
    ax.axvline(24, color=TOKENS["gold"], linestyle="--", linewidth=1.4)
    ax.text(24.1, ax.get_ylim()[1] * 0.96, "24h limit", color=TOKENS["gold"], fontsize=9, va="top")
    ax.set_xlabel("Runtime since W&B start (hours)")
    ax.set_ylabel("Logged training step (millions)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    style_ax(ax)
    add_chart_header(
        fig,
        ax,
        "Cumulative progress shows a persistent throughput gap",
        "W&B _runtime versus _step; e9esa15p reaches 1.71M steps by timeout, y5a0upzd reaches 2.40M in 19.0h.",
    )
    path = ASSET_DIR / "progress_trend.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    chart_paths["progress"] = path.name

    # Bar comparison: observed and projected throughput.
    fig, ax = plt.subplots(figsize=(8.4, 4.9), dpi=160, facecolor=TOKENS["surface"])
    plot_df = summaries.copy()
    plot_df["steps_per_h_k"] = plot_df["steps_per_h"] / 1000
    colors = [TOKENS["orange"] if rid == "e9esa15p" else TOKENS["blue"] for rid in plot_df["run_id"]]
    bars = ax.barh(plot_df["label"], plot_df["steps_per_h_k"], color=colors, edgecolor=TOKENS["ink"], linewidth=0.4)
    ax.bar_label(bars, fmt="%.1fk", padding=4, fontsize=9, color=TOKENS["ink"])
    ax.set_xlabel("Average logged steps per hour")
    ax.set_ylabel("")
    style_ax(ax)
    add_chart_header(
        fig,
        ax,
        "The timeout run is about 44% slower per logged step",
        "Average W&B throughput over the whole run, including validation/checkpoint overhead.",
    )
    path = ASSET_DIR / "throughput_bar.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    chart_paths["throughput"] = path.name

    # Windowed speed chart: 100k-step windows.
    window_frames = []
    for run_id, bundle in bundles.items():
        df = bundle.history
        last = int(df["_step"].max() // 100_000 * 100_000)
        points = list(range(100_000, last + 1, 100_000))
        prev_step = 0
        prev_runtime = runtime_at_step(df, 0)
        for point in points:
            rt = runtime_at_step(df, point)
            if rt > prev_runtime:
                window_frames.append(
                    {
                        "run": RUN_LABELS[run_id],
                        "run_id": run_id,
                        "step_m": point / 1_000_000,
                        "steps_per_s": (point - prev_step) / (rt - prev_runtime),
                    }
                )
            prev_step, prev_runtime = point, rt
    windows = pd.DataFrame(window_frames)
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=160, facecolor=TOKENS["surface"])
    for label, group in windows.groupby("run"):
        ax.plot(
            group["step_m"],
            group["steps_per_s"],
            color=palette[label],
            marker="o",
            linewidth=1.8,
            markersize=3.5,
            label=label,
        )
    ax.set_xlabel("Training step reached (millions)")
    ax.set_ylabel("Steps per second in previous 100k-step window")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=9)
    style_ax(ax)
    add_chart_header(
        fig,
        ax,
        "The gap is not a late-only failure",
        "100k-step windows from W&B runtime interpolation; e9esa15p stays below the completed run across the overlapping range.",
    )
    path = ASSET_DIR / "window_speed.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    chart_paths["window_speed"] = path.name

    return chart_paths


def fmt(value: Any, digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    try:
        if math.isnan(float(value)):
            return "n/a"
    except (TypeError, ValueError):
        return str(value)
    return f"{float(value):,.{digits}f}{suffix}"


def html_table(df: pd.DataFrame, columns: list[str]) -> str:
    return df[columns].to_html(index=False, classes="data-table", border=0, escape=False)


def build_html(
    bundles: dict[str, RunBundle],
    summaries: pd.DataFrame,
    system_summaries: dict[str, dict[str, Any]],
    chart_paths: dict[str, str],
    config_diff: list[dict[str, Any]],
) -> None:
    e9 = summaries.set_index("run_id").loc["e9esa15p"]
    y5 = summaries.set_index("run_id").loc["y5a0upzd"]
    speed_ratio = e9["steps_per_s"] / y5["steps_per_s"]
    e9_projected_h = e9["projected_runtime_h_for_target"]
    y5_to_2m_h = 2_000_000 / y5["steps_per_s"] / 3600

    summary_cols = [
        "Run",
        "State",
        "Target steps",
        "Last step",
        "Runtime",
        "Steps/hour",
        "Episodes/hour",
        "SLURM state",
        "Node/GPU",
    ]
    table_df = summaries.assign(
        Run=lambda x: x["run_id"].map(RUN_LABELS),
        State=lambda x: x["state"],
        **{
            "Target steps": lambda x: x["target_steps"].map(lambda v: fmt(v, 0)),
            "Last step": lambda x: x["last_step"].map(lambda v: fmt(v, 0)),
            "Runtime": lambda x: x["runtime_h"].map(lambda v: fmt(v, 2, "h")),
            "Steps/hour": lambda x: x["steps_per_h"].map(lambda v: fmt(v / 1000, 1, "k")),
            "Episodes/hour": lambda x: x["episodes_per_h"].map(lambda v: fmt(v, 0)),
            "SLURM state": lambda x: x["slurm_state"].fillna("n/a"),
            "Node/GPU": lambda x: x.apply(lambda r: f"{r['node'] or 'n/a'} / {r['gpu'] or 'n/a'}", axis=1),
        },
    )

    config_df = pd.DataFrame(config_diff)
    important_diff = config_df[config_df["key"].isin(["total_steps", "seed", "logdir"])]
    if important_diff.empty:
        important_diff = config_df.head(8)

    system_rows = []
    for rid, metrics in system_summaries.items():
        row = {"run_id": rid, "Run": RUN_LABELS[rid]}
        for key, label in [
            ("system.gpu.0.gpu", "GPU util median"),
            ("system.gpu.0.memory", "GPU memory median"),
            ("system.gpu.0.memoryAllocatedBytes", "GPU allocated median GB"),
            ("system.cpu", "CPU median"),
            ("system.proc.memory.rssMB", "RSS median MB"),
        ]:
            if key in metrics:
                row[label] = fmt(metrics[key]["median"], 1)
            else:
                row[label] = "not logged"
        system_rows.append(row)
    system_df = pd.DataFrame(system_rows)

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>W&B runtime comparison: e9esa15p vs y5a0upzd</title>
  <style>
    :root {{
      --surface: #FCFCFD;
      --panel: #FFFFFF;
      --ink: #1F2430;
      --muted: #6F768A;
      --grid: #E6E8F0;
      --axis: #D7DBE7;
      --blue: #5477C4;
      --orange: #CC6F47;
      --gold: #B8A037;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--surface);
      color: var(--ink);
      font-family: Inter, Aptos, "Segoe UI", Arial, sans-serif;
      line-height: 1.55;
    }}
    main {{
      max-width: 1040px;
      margin: 0 auto;
      padding: 40px 24px 64px;
    }}
    h1 {{ font-size: 2rem; margin: 0 0 8px; letter-spacing: 0; }}
    h2 {{ font-size: 1.35rem; margin: 40px 0 12px; letter-spacing: 0; }}
    h3 {{ font-size: 1.05rem; margin: 24px 0 8px; letter-spacing: 0; }}
    p {{ margin: 0 0 14px; }}
    .muted {{ color: var(--muted); }}
    .summary {{
      border-left: 4px solid var(--orange);
      padding: 12px 0 12px 18px;
      margin: 20px 0 24px;
      background: #fff;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin: 24px 0 28px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--grid);
      border-radius: 8px;
      padding: 14px 14px 12px;
    }}
    .metric strong {{
      display: block;
      font-size: 1.35rem;
      line-height: 1.2;
      margin-bottom: 4px;
    }}
    .metric span {{ color: var(--muted); font-size: 0.9rem; }}
    figure {{
      margin: 24px 0;
      background: var(--panel);
      border: 1px solid var(--grid);
      border-radius: 8px;
      padding: 12px;
    }}
    figure img {{ width: 100%; display: block; }}
    figcaption {{ color: var(--muted); font-size: 0.9rem; margin: 8px 4px 2px; }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      margin: 14px 0 22px;
      font-size: 0.94rem;
      background: var(--panel);
      border: 1px solid var(--grid);
    }}
    .data-table th, .data-table td {{
      border-bottom: 1px solid var(--grid);
      padding: 9px 10px;
      text-align: left;
      vertical-align: top;
    }}
    .data-table th {{ color: var(--muted); font-weight: 650; background: #F7F8FB; }}
    code {{
      font-family: "SF Mono", Menlo, Consolas, monospace;
      background: #F4F5F7;
      padding: 0.1rem 0.28rem;
      border-radius: 4px;
    }}
    ul {{ padding-left: 1.25rem; }}
    li {{ margin: 0.35rem 0; }}
    a {{ color: var(--blue); }}
    @media (max-width: 760px) {{
      main {{ padding: 28px 16px 48px; }}
      .metric-grid {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 1.6rem; }}
    }}
  </style>
</head>
<body>
<main>
  <h1>W&amp;B runtime comparison: e9esa15p vs y5a0upzd</h1>
  <p class="muted">Generated {generated}. Sources: W&amp;B API, local metrics.csv files, local SLURM logs, and local MANIFEST.json where available.</p>

  <section class="summary">
    <p><strong>Answer:</strong> <code>e9esa15p</code> did not miss completion because of a small tail-end issue. It was materially slower throughout the run and then hit the 24h SLURM limit. W&amp;B shows {fmt(e9['steps_per_s'], 2)} steps/s for <code>e9esa15p</code> versus {fmt(y5['steps_per_s'], 2)} steps/s for <code>y5a0upzd</code>, so the timeout run ran at only {fmt(speed_ratio * 100, 0)}% of the completed run's step throughput.</p>
    <p>At its observed pace, <code>e9esa15p</code> needed about {fmt(e9_projected_h, 1)}h to reach its own 2.0M-step target. The completed run's pace would have reached 2.0M steps in about {fmt(y5_to_2m_h, 1)}h and actually reached 2.4M steps in {fmt(y5['runtime_h'], 1)}h.</p>
  </section>

  <div class="metric-grid">
    <div class="metric"><strong>{fmt(e9['runtime_h'], 1)}h</strong><span>e9esa15p runtime before timeout</span></div>
    <div class="metric"><strong>{fmt(e9['last_step'] / 1_000_000, 2)}M</strong><span>e9esa15p last logged step</span></div>
    <div class="metric"><strong>{fmt(y5['last_step'] / 1_000_000, 2)}M</strong><span>y5a0upzd completed steps</span></div>
    <div class="metric"><strong>{fmt((1 - speed_ratio) * 100, 0)}%</strong><span>average step-throughput gap</span></div>
  </div>

  <h2>The two runs were not parity-identical</h2>
  <p>The visible architecture-level settings are close enough to compare as CNN L1 training, but the run setup is not exactly the same. The June run is from <code>{e9['git_branch'] or 'unknown branch'}</code> at <code>{e9['git_sha'] or 'unknown sha'}</code> with a dirty manifest and <code>total_steps=2,000,000</code>. The April run is the older buffer-fix baseline with <code>total_steps=2,400,000</code> and no local manifest captured in this checkout. The SLURM logs also show different nodes and driver stacks: <code>{e9['node']}</code> / {e9['gpu']} versus <code>{y5['node']}</code> / {y5['gpu']}.</p>
  {html_table(table_df, summary_cols)}

  <h2>The throughput gap is visible in W&amp;B runtime</h2>
  <p>The most direct comparison is W&amp;B <code>_runtime</code> against logged training step. This avoids relying on log verbosity or checkpoint timestamps. The completed baseline advances much faster over the same wall-clock window; by 19h it has completed 2.4M steps, while the timeout run reaches only about 1.35M steps by the same runtime and 1.71M by 24h.</p>
  <figure>
    <img src="wandb_runtime_e9esa15p_y5a0upzd_assets/{chart_paths['progress']}" alt="Cumulative W&B progress trend">
    <figcaption>Trend chart: W&amp;B runtime in hours versus logged training step in millions.</figcaption>
  </figure>

  <p>Average throughput over the full recorded run is the simplest numerical summary: <code>e9esa15p</code> logged {fmt(e9['steps_per_h'] / 1000, 1)}k steps/hour, while <code>y5a0upzd</code> logged {fmt(y5['steps_per_h'] / 1000, 1)}k steps/hour.</p>
  <figure>
    <img src="wandb_runtime_e9esa15p_y5a0upzd_assets/{chart_paths['throughput']}" alt="Average throughput comparison">
    <figcaption>Bar chart: average W&amp;B step throughput across each run.</figcaption>
  </figure>

  <h2>The slowdown is sustained, not just a final crash</h2>
  <p>Windowed 100k-step throughput stays lower for the June run across the overlapping training range. This points away from a single final checkpoint, W&amp;B sync, or crash-only explanation. The decisive issue is lower steady-state progress.</p>
  <figure>
    <img src="wandb_runtime_e9esa15p_y5a0upzd_assets/{chart_paths['window_speed']}" alt="Windowed speed comparison">
    <figcaption>Line chart: steps per second in the previous 100k-step window.</figcaption>
  </figure>

  <h2>Validation/checkpointing does not explain the whole gap</h2>
  <p>The April baseline logs periodic validation and completes final validation/checkpointing. The June timeout run has fewer synced files in W&amp;B and the SLURM log ends at the scheduler timeout, but the W&amp;B runtime curve is already slower before the end. The conclusion is that validation/checkpointing may contribute to overhead, but it is not the primary cause of the 24h miss.</p>

  <h2>System metrics suggest environment/runtime differences, not a model-size change</h2>
  <p>Both jobs report H100-class GPUs and the same 8 CPU / 64 GB allocation. However, the runs are on different nodes, driver versions, W&amp;B versions, and branch/runtime environments. The June job log also starts by uninstalling and reinstalling 17 packages, which is not present in the April run log and indicates the environment was not operationally identical.</p>
  {html_table(system_df, ["Run", "GPU util median", "GPU memory median", "GPU allocated median GB", "CPU median", "RSS median MB"])}

  <h2>Important config differences</h2>
  <p>The W&amp;B config diff is small but enough to reject the assumption that the runs are exact duplicates. The target step count differs, the seed differs, and the June run records newer encoder/VGGT-related config keys that the April run did not log.</p>
  {html_table(important_diff.rename(columns={"key": "Config key", "e9esa15p": "e9esa15p", "y5a0upzd": "y5a0upzd"}), ["Config key", "e9esa15p", "y5a0upzd"])}

  <h2>Conclusion</h2>
  <ul>
    <li><strong>Primary cause:</strong> <code>e9esa15p</code> has much lower steady-state step throughput, about {fmt((1 - speed_ratio) * 100, 0)}% below <code>y5a0upzd</code>.</li>
    <li><strong>Immediate failure mode:</strong> SLURM killed job <code>4974622</code> at the 24h timelimit, leaving the run at {fmt(e9['completion_pct'], 1)}% of its 2.0M-step target.</li>
    <li><strong>Parity finding:</strong> The setup is not exactly the same: branch/SHA, target steps, seed, node, driver, W&amp;B version, and package setup differ.</li>
    <li><strong>Most likely investigation path:</strong> reproduce the June branch/run config on the faster node class or rerun a short A/B smoke on both nodes with fixed SHA, seed, target, validation cadence, and no environment mutation at startup. The key metric should be W&amp;B <code>_step/_runtime</code> over the first 200k-500k steps.</li>
  </ul>

  <h2>Audit notes</h2>
  <p>Local evidence paths used by the generator: <code>output/r2dreamer-l1-replay-capacity/cnn-cap1m-seed4367942/</code>, <code>output/runs/r2dreamer-curriculum-l1-rerun/</code>, W&amp;B runs <a href="{bundles['e9esa15p'].run.url}">e9esa15p</a> and <a href="{bundles['y5a0upzd'].run.url}">y5a0upzd</a>. The analysis treats W&amp;B <code>_step</code> as the logged training/environment-step axis; "episodes" in the user prompt appears to refer to million-step progress, because W&amp;B episode counts are 6,362 and 9,105, not millions.</p>
</main>
</body>
</html>
"""
    REPORT_PATH.write_text(html)


def main() -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    api = wandb.Api(timeout=90)
    bundles: dict[str, RunBundle] = {}
    config_by_run: dict[str, dict[str, Any]] = {}
    system_summaries: dict[str, dict[str, Any]] = {}

    for run_id in RUN_IDS:
        run = api.run(f"{PROJECT}/{run_id}")
        keys = history_keys(run)
        history = dedupe_step_runtime(scan(run, ["_step", "_runtime", "_timestamp"]))
        episodes_keys = ["_step", "_runtime", "episode/count", "episode/steps", "metrics/sr", "metrics/spl"]
        episodes = scan(run, [k for k in episodes_keys if k in keys or k.startswith("_")])
        system = scan_system_metrics(run)
        metadata = load_wandb_metadata(run, run_id)
        manifest = read_json(LOCAL_PATHS[run_id].get("manifest", Path("/missing")))
        slurm = parse_slurm(run_id)
        bundles[run_id] = RunBundle(run_id, run, history, episodes, system, metadata, manifest, slurm)
        config_by_run[run_id] = dict(run.config)
        system_summaries[run_id] = summarize_system(system, run_id)

    summaries = pd.DataFrame([summarize_bundle(bundle) for bundle in bundles.values()])
    summaries["label"] = summaries["run_id"].map(RUN_LABELS)
    summaries.to_csv(ASSET_DIR / "run_summary.csv", index=False)

    all_config_keys = sorted(set(config_by_run[RUN_IDS[0]]) | set(config_by_run[RUN_IDS[1]]))
    config_diff = []
    for key in all_config_keys:
        left = config_by_run[RUN_IDS[0]].get(key, "<missing>")
        right = config_by_run[RUN_IDS[1]].get(key, "<missing>")
        if left != right:
            config_diff.append({"key": key, RUN_IDS[0]: repr(left), RUN_IDS[1]: repr(right)})
    pd.DataFrame(config_diff).to_csv(ASSET_DIR / "config_diff.csv", index=False)
    (ASSET_DIR / "system_summary.json").write_text(json.dumps(system_summaries, indent=2, sort_keys=True))

    chart_paths = build_charts(bundles, summaries)
    build_html(bundles, summaries, system_summaries, chart_paths, config_diff)

    print(f"Wrote {REPORT_PATH}")
    print(summaries[["run_id", "state", "last_step", "runtime_h", "steps_per_s", "projected_runtime_h_for_target"]].to_string(index=False))


if __name__ == "__main__":
    main()
