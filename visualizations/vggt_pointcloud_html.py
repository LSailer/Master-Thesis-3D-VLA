"""Render VGGT world_points to an interactive HTML scatter plot.

Usage:
  python -m visualizations.vggt_pointcloud_html --npz output/points.npz --out output/points.html

NPZ must contain:
  - world_points: (37, 37, 3) float32
Optional:
  - camera_pose: (9,) float32 (unused for now)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>VGGT Point Cloud</title>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
</head>
<body>
  <div id="plot" style="width: 100%; height: 100vh;"></div>
  <script>
    const x = {x};
    const y = {y};
    const z = {z};
    const colors = {c};

    const trace = {
      x, y, z,
      mode: 'markers',
      type: 'scatter3d',
      marker: {
        size: 2,
        color: colors,
        colorscale: 'Viridis',
        opacity: 0.9
      }
    };

    const layout = {
      margin: {l: 0, r: 0, b: 0, t: 30},
      title: 'VGGT world_points (37x37)',
      scene: {
        aspectmode: 'data',
        xaxis: {title: 'x'},
        yaxis: {title: 'y'},
        zaxis: {title: 'z'}
      }
    };

    Plotly.newPlot('plot', [trace], layout, {responsive: true});
  </script>
</body>
</html>
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", required=True, type=Path, help="Input npz with world_points")
    p.add_argument("--out", required=True, type=Path, help="Output HTML path")
    args = p.parse_args()

    data = np.load(args.npz)
    world_points = data["world_points"]  # (37,37,3)
    pts = world_points.reshape(-1, 3)
    x = pts[:, 0].tolist()
    y = pts[:, 1].tolist()
    z = pts[:, 2].tolist()

    # Color by depth (z).
    colors = pts[:, 2].tolist()

    html = HTML_TEMPLATE.format(
        x=json.dumps(x),
        y=json.dumps(y),
        z=json.dumps(z),
        c=json.dumps(colors),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
