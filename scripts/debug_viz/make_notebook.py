"""Build notebooks/debug_viz_l1.ipynb from a list of cell sources.

Run once to produce the .ipynb. The notebook itself just loads the bundles
prepared by render_notebook_data.py and renders inline plotly + matplotlib.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def md(text: str):
    """Create a Markdown notebook cell from text with surrounding newlines stripped."""
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str):
    """Create a code notebook cell from text with surrounding newlines stripped."""
    return nbf.v4.new_code_cell(text.strip("\n"))


CELLS = [
    md("""
# R2Dreamer encoder-drift on L1 VGGT (ckpt 300k) — interactive viz

Companion to [`docs/wiki/methods/r2dreamer-encoder-drift-viz.md`](../docs/wiki/methods/r2dreamer-encoder-drift-viz.md).

Pair: **ep7** (clean success, 135 steps, SPL 0.96) vs **ep1** (near-miss, 500 steps, ends 0.94 m from goal — **4.7× outside** the 0.2 m success radius; a final-approach failure, not a near-miss inside the goal).

This notebook is a thin viewer of pre-rendered artifacts. Run order top-to-bottom; each cell is independent given the bundle files.
"""),

    code("""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import Image, display

REPO = Path.cwd()
while REPO.name and not (REPO / "AGENTS.md").exists():
    REPO = REPO.parent
NOTEBOOK_DIR = REPO / "output/methods/debug_viz/l1/notebook"
SIMILARITY_DIR = REPO / "output/methods/debug_viz/l1/similarity"

def _load_bundle(ep_idx: int):
    b = np.load(NOTEBOOK_DIR / f"bundle_ep{ep_idx:03d}.npz", allow_pickle=False)
    meta = json.loads(str(b["meta_json"]))
    return {**{k: b[k] for k in b.files if k != "meta_json"}, "meta": meta}

ep7 = _load_bundle(7)
ep1 = _load_bundle(1)
print(f"ep7: T={ep7['T']}, frames sampled at {list(ep7['chosen_frame_indices'])}, success={ep7['meta']['success']}")
print(f"ep1: T={ep1['T']}, frames sampled at {list(ep1['chosen_frame_indices'])}, success={ep1['meta']['success']}")
"""),

    md("""
## 1. Summary table

The three coherent drift signals at a glance.
"""),

    code("""
import pandas as pd

def _row(b):
    iu = np.triu_indices(b["S_VGGT"].shape[0], k=1)
    return {
        "T": b["T"],
        "success": b["meta"]["success"],
        "spl": round(b["meta"]["spl"], 3),
        "probe_feat RMSE (m)": round(float(np.sqrt(b["mse_feat"]).mean()), 3),
        "probe_deter RMSE (m)": round(float(np.sqrt(b["mse_deter"]).mean()), 3),
        "S_VGGT off-diag mean": round(float(b["S_VGGT"][iu].mean()), 3),
        "S_feat off-diag mean": round(float(b["S_feat"][iu].mean()), 3),
        "S_deter off-diag mean": round(float(b["S_deter"][iu].mean()), 3),
    }

df = pd.DataFrame({"ep7 (success)": _row(ep7), "ep1 (near-miss)": _row(ep1)})
df
"""),

    md("""
## 2. Temporal similarity matrices C2 (pre-rendered PNG)

5-panel per episode: `S_VGGT | S_feat | S_deter | diff(VGGT−feat) | diff(VGGT−deter)`.

ep1's `S_feat` and `S_deter` look more uniform (whitewashed) than `S_VGGT` — Dreamer compresses the geometric diversity that VGGT correctly distinguishes. That's the representation-collapse signal.
"""),

    code("""
display(Image(filename=str(SIMILARITY_DIR / "ep_007/similarity.png"), width=1100))
display(Image(filename=str(SIMILARITY_DIR / "ep_001/similarity.png"), width=1100))
"""),

    md("""
## 3. Probe-error timeline C1

RMSE per step for each of the three probes. Note the Y-axis difference between the two episodes.
"""),

    code("""
display(Image(filename=str(SIMILARITY_DIR / "ep_007/probe_error_timeline.png"), width=850))
display(Image(filename=str(SIMILARITY_DIR / "ep_001/probe_error_timeline.png"), width=850))
"""),

    md("""
## 4. Interactive 3D scene + agent trajectory

Five sampled frames per episode (start / 25% / 50% / 75% / end−1). Each frame's VGGT cloud shown in 3D, agent trajectory overlaid, agent's position at the chosen frame highlighted. Use the dropdown above each plot to switch frames.
"""),

    code("""
def _plot_3d_with_trajectory(b, title):
    traj = b["trajectory"]   # (T, 3) world coords
    frames_idx = b["chosen_frame_indices"]
    clouds = b["chosen_world_points"]  # (K, 37, 37, 3)
    K = len(frames_idx)

    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scatter3d"}]])

    # Agent trajectory line, always visible (Habitat: x=east, y=up, z=north)
    fig.add_trace(go.Scatter3d(
        x=traj[:, 0], y=traj[:, 2], z=traj[:, 1],  # swap y/z so up is up
        mode="lines",
        line=dict(width=4, color="rgba(40,40,40,0.85)"),
        name="agent trajectory",
        hoverinfo="skip",
    ))

    # Goal positions
    goals = np.array(b["meta"]["goal_positions"])
    fig.add_trace(go.Scatter3d(
        x=goals[:, 0], y=goals[:, 2], z=goals[:, 1],
        mode="markers",
        marker=dict(size=7, color="gold", symbol="diamond"),
        name="goal positions",
    ))

    # Per-frame: cloud (camera-frame, NOT world frame — relative geometry only)
    cloud_traces = []
    for k, t in enumerate(frames_idx):
        # cloud is in CAMERA frame per VGGT.  Translate it to the agent's
        # world position so the visualization is human-readable. Rotation by
        # quaternion would be more correct but we skip for clarity.
        wp = clouds[k].reshape(-1, 3) + traj[t]
        depth = clouds[k].reshape(-1, 3)[:, 2]  # camera-z = depth
        cloud_traces.append(go.Scatter3d(
            x=wp[:, 0], y=wp[:, 2], z=wp[:, 1],
            mode="markers",
            marker=dict(size=1.6, color=depth, colorscale="Viridis", showscale=False, opacity=0.6),
            name=f"VGGT cloud frame {t}",
            visible=(k == 0),
        ))
        cloud_traces.append(go.Scatter3d(
            x=[traj[t, 0]], y=[traj[t, 2]], z=[traj[t, 1]],
            mode="markers",
            marker=dict(size=8, color="red", symbol="circle"),
            name=f"agent @ frame {t}",
            visible=(k == 0),
        ))
    for tr in cloud_traces:
        fig.add_trace(tr)

    # Visibility toggles per frame
    n_static = 2  # trajectory + goals
    buttons = []
    for k in range(K):
        vis = [True, True]  # static traces
        for kk in range(K):
            vis += [kk == k, kk == k]  # cloud + agent dot
        buttons.append(dict(label=f"frame {int(frames_idx[k])}", method="update", args=[{"visible": vis}]))

    fig.update_layout(
        title=title,
        height=620,
        scene=dict(
            xaxis_title="x (m)", yaxis_title="z (m, north)", zaxis_title="y (m, up)",
            aspectmode="data",
        ),
        updatemenus=[dict(type="dropdown", showactive=True, buttons=buttons, x=0.02, y=0.98)],
        legend=dict(orientation="h", x=0, y=-0.05),
    )
    fig.show()

_plot_3d_with_trajectory(ep7, "ep7 (clean success, T=135) — VGGT cloud + agent trajectory")
"""),

    code("""
_plot_3d_with_trajectory(ep1, "ep1 (near-miss, T=500) — VGGT cloud + agent trajectory")
"""),

    md("""
## 5. VGGT vs probe reconstruction (one diagnostic frame each)

Per episode, render the **true VGGT cloud** next to the **probe_feat predicted cloud** for the middle sampled frame. If the probe captures geometry, the two clouds look similar; if the encoder has dropped geometry, the predicted cloud is degraded.

The probe is post-hoc Ridge `feat (2560) → world_points (4107)` trained on the other 10 episodes. RMSE per frame shown on the right.
"""),

    code("""
def _plot_recon_pair(b, title_prefix):
    # Pick the middle of the 5 sampled frames
    k = 2
    t = int(b["chosen_frame_indices"][k])
    true_cloud = b["chosen_world_points"][k].reshape(-1, 3)
    pred_cloud = b["pred_world_points_feat"][k].reshape(-1, 3)
    err = np.linalg.norm(true_cloud - pred_cloud, axis=1)  # per-point error in m

    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
                        subplot_titles=[f"VGGT (true)", f"probe_feat predicted (per-point err in m)"])
    fig.add_trace(go.Scatter3d(
        x=true_cloud[:, 0], y=true_cloud[:, 2], z=true_cloud[:, 1],
        mode="markers",
        marker=dict(size=2, color=true_cloud[:, 2], colorscale="Viridis", showscale=False, opacity=0.7),
        name="VGGT true",
    ), row=1, col=1)
    fig.add_trace(go.Scatter3d(
        x=pred_cloud[:, 0], y=pred_cloud[:, 2], z=pred_cloud[:, 1],
        mode="markers",
        marker=dict(size=2, color=err, colorscale="Reds", cmin=0, cmax=max(0.5, float(err.max())),
                    colorbar=dict(title="err (m)", x=1.02, y=0.5, len=0.7), opacity=0.7),
        name="probe_feat pred",
    ), row=1, col=2)
    rmse = float(np.sqrt((err ** 2).mean()))
    fig.update_layout(
        title=f"{title_prefix} | frame {t} | per-frame RMSE = {rmse:.3f} m",
        height=520,
        scene=dict(aspectmode="data"),
        scene2=dict(aspectmode="data"),
    )
    fig.show()

_plot_recon_pair(ep7, "ep7 (success)")
_plot_recon_pair(ep1, "ep1 (near-miss)")
"""),

    md("""
## 6. Reading guide & next steps

**What the figures show together**

- §1 numerics: `probe_feat` reconstructs ep7 at RMSE 0.23 m vs ep1 at 0.37 m. Off-diag cosine mean of `S_feat` is 0.725 on ep7 and 0.771 on ep1 — Dreamer flattens ep1's similarity structure compared to VGGT.
- §2 panels: ep1's `S_feat` and `S_deter` look noticeably more uniform than `S_VGGT`. Diff matrices show where Dreamer disagrees most.
- §3 timelines: probe RMSE per step. Watch for monotonic growth (compounding drift) vs localized spikes (transient drift).
- §4 3D scenes: spatial intuition for the trajectory. ep1 covers more of the room than ep7 yet Dreamer compresses it more in latent space.
- §5 reconstructions: side-by-side check whether the probe-predicted cloud preserves coarse geometry. Limited by R²; intended as illustration not measurement.

**Open**

- Replicate on a second pair (e.g. ep10 + ep0) to move past n=1 per regime.
- If the figure story under-delivers, escalate from linear ridge probe to small MLP. Wiki page §Limitations explains why this likely won't change the *gap* signal.
- Actor-termination hypothesis: ep1 ends inside the 1 m radius without STOP. A focused experiment could pull `actor.apply(feat)` logits at every step of ep1 and inspect whether STOP probability ever spikes.
"""),
]


def main() -> None:
    """Write the debug visualization notebook from the static cell list."""
    nb = nbf.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    out = Path("notebooks/debug_viz_l1.ipynb")
    out.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(out))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
