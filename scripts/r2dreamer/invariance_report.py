"""Render the VGGT first-frame invariance findings as a self-contained HTML report.

Pure NumPy / Matplotlib — consumes the .npz artifact produced by
invariance_extract.py. Used both by the analysis notebook and as a CLI:

    python scripts/r2dreamer/invariance_report.py \
        --artifact-dir output/analysis/invariance \
        --out docs/3d-invariance-vggt-first-frame.html
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.analysis.invariance_metrics import umeyama, match_by_position, compare_pair  # noqa: E402


def _fig_b64(fig) -> str:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _rgb_b64(arr: np.ndarray) -> str:
    """Encode an (H,W,3) uint8 array as a base64 PNG via matplotlib."""
    import matplotlib
    img = matplotlib.image.imsave
    buf = io.BytesIO()
    img(buf, np.ascontiguousarray(arr), format="png")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def load(artifact_dir: Path):
    artifact_dir = Path(artifact_dir)
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    eps = []
    for e in manifest["episodes"]:
        d = np.load(artifact_dir / e["file"])
        eps.append({k: d[k] for k in d.files})
    return manifest, eps


def compute(eps, match_max_dist=0.30):
    rows, matches = [], []
    for i, ep in enumerate(eps):
        pairs, dists = match_by_position(ep["fwd_pos"], ep["bwd_pos"], match_max_dist)
        matches.append((pairs, dists))
        for (f, b) in pairs:
            m = compare_pair(ep["fwd_wp"][f], ep["bwd_wp"][b], ep["fwd_cp"][f],
                             ep["bwd_cp"][b], ep["fwd_logit"][f], ep["bwd_logit"][b],
                             ep["fwd_deter"][f], ep["bwd_deter"][b],
                             ep["fwd_embed"][f], ep["bwd_embed"][b])
            m.update(episode=i, f=f, b=b)
            rows.append(m)
    return rows, matches


def _agg(rows, key):
    v = np.array([r[key] for r in rows], dtype=np.float64)
    return float(v.mean()), float(v.std()), float(v.min()), float(v.max())


def build_report(artifact_dir, out_path, match_max_dist=0.30):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    artifact_dir, out_path = Path(artifact_dir), Path(out_path)
    manifest, eps = load(artifact_dir)
    rows, matches = compute(eps, match_max_dist)
    if not rows:
        raise RuntimeError("No matched frame pairs — check match_max_dist / paths.")

    # ---- per-frame curves --------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    episs = sorted({r["episode"] for r in rows})
    for i in episs:
        s = sorted([r for r in rows if r["episode"] == i], key=lambda r: r["f"])
        xs = [r["f"] for r in s]
        ax[0, 0].plot(xs, [r["wp_cosine"] for r in s], marker="o", ms=3, label=f"ep{i}")
        ax[0, 1].plot(xs, [r["wp_rmse_raw"] for r in s], marker="o", ms=3)
        ax[0, 1].plot(xs, [r["wp_rmse_aligned"] for r in s], marker="x", ms=3, ls="--")
        ax[1, 0].plot(xs, [r["latent_js"] for r in s], marker="o", ms=3)
        ax[1, 1].plot(xs, [r["deter_cosine"] for r in s], marker="o", ms=3)
    ax[0, 0].set(title="Raw WP cosine (fwd vs bwd)", xlabel="forward frame", ylabel="cosine")
    ax[0, 1].set(title="WP RMSE: raw (solid) vs Umeyama-aligned (dashed)",
                 xlabel="forward frame", ylabel="RMSE (m)")
    ax[1, 0].set(title="RSSM posterior JS divergence", xlabel="forward frame", ylabel="JS (nats)")
    ax[1, 1].set(title="RSSM deterministic-state cosine", xlabel="forward frame", ylabel="cosine")
    ax[0, 0].legend(fontsize=8)
    for a in ax.ravel():
        a.grid(alpha=.3)
    fig.tight_layout()
    curves_b64 = _fig_b64(fig)

    # ---- WP point clouds for one pair -------------------------------------
    (pairs0, _) = matches[0]
    f, b = pairs0[len(pairs0) // 2]
    fwp = eps[0]["fwd_wp"][f].reshape(-1, 3)
    bwp = eps[0]["bwd_wp"][b].reshape(-1, 3)
    al = umeyama(bwp, fwp, with_scale=True)
    fig = plt.figure(figsize=(13, 5.5))
    a1 = fig.add_subplot(121, projection="3d")
    a1.scatter(*fwp.T, s=3, alpha=.4, label="forward")
    a1.scatter(*bwp.T, s=3, alpha=.4, label="backward (raw)")
    a1.set_title(f"Raw WP — ep0 fwd#{f} vs bwd#{b}\nRMSE={al['rmse_before']:.3f} m")
    a1.legend()
    a2 = fig.add_subplot(122, projection="3d")
    a2.scatter(*fwp.T, s=3, alpha=.4, label="forward")
    a2.scatter(*al["aligned"].T, s=3, alpha=.4, label="backward (aligned)")
    a2.set_title(f"After Umeyama — residual={al['rmse_after']:.3f} m (s={al['s']:.3f})")
    a2.legend()
    fig.tight_layout()
    cloud_b64 = _fig_b64(fig)

    # ---- matched RGB thumbnails (same place, opposite heading) ------------
    def chw_or_hwc(a):
        return np.transpose(a, (1, 2, 0)) if a.shape[0] == 3 else a
    rgb_f = _rgb_b64(chw_or_hwc(eps[0]["fwd_rgb"][f]))
    rgb_b = _rgb_b64(chw_or_hwc(eps[0]["bwd_rgb"][b]))

    # ---- aggregates --------------------------------------------------------
    agg_keys = ["wp_cosine", "wp_rmse_raw", "wp_rmse_aligned", "wp_residual_ratio",
                "cp_l2", "latent_js", "deter_cosine"]
    if "embed_cosine" in rows[0]:
        agg_keys.insert(4, "embed_cosine")
    agg = {k: _agg(rows, k) for k in agg_keys}

    rr_mean = agg["wp_residual_ratio"][0]
    js_mean = agg["latent_js"][0]
    frame_artefact = rr_mean < 0.35
    model_sensitive = js_mean > 0.05

    verdict = []
    verdict.append(
        f"Raw WP clouds disagree by <b>{agg['wp_rmse_raw'][0]:.3f} m</b> RMSE on average, "
        f"but after a rigid+scale (Umeyama) transform the residual drops to "
        f"<b>{agg['wp_rmse_aligned'][0]:.3f} m</b> "
        f"(residual ratio {rr_mean:.2f}). ")
    if frame_artefact:
        verdict.append(
            "The two runs therefore encode essentially the <b>same geometry in different "
            "reference frames</b> — the divergence is a first-frame-relative artefact, "
            "confirming the VGGT invariance concern at the input level.")
    else:
        verdict.append(
            "A large residual remains even after alignment, so the runs disagree "
            "<b>structurally</b>, not merely by a reference-frame transform.")
    verdict.append(
        f" The trained Dreamer posterior diverges by JS = <b>{js_mean:.3f} nats</b> "
        f"(deter cosine {agg['deter_cosine'][0]:.3f}). ")
    verdict.append(
        "This indicates the 2M-step model is <b>not invariant</b> to the start pose: the "
        "first-frame-relative encoding propagates into the latent state."
        if model_sensitive else
        "This is small, suggesting the model has substantially <b>absorbed</b> the "
        "reference-frame variation into a stable latent state.")

    rows_html = "".join(
        f"<tr><td>{k}</td><td>{m:.4f}</td><td>{s:.4f}</td><td>{lo:.4f}</td><td>{hi:.4f}</td></tr>"
        for k, (m, s, lo, hi) in agg.items())
    ep_html = "".join(
        f"<tr><td>{e['index']}</td><td>{e['episode_id']}</td><td>{e['geodesic']:.2f}</td>"
        f"<td>{e['fwd_steps']}</td><td>{e['bwd_steps']}</td>"
        f"<td>{len(matches[e['index']][0])}</td></tr>"
        for e in manifest["episodes"])

    mode = manifest.get("mode", "navigate")
    if mode == "teleport":
        mode_note = ("<b>Mode: teleport-replay.</b> The backward run revisits the exact "
                     "forward camera poses (same position <i>and</i> heading) in reverse with "
                     "a fresh VGGT window, so the only variable is the first-frame reference — "
                     "the heading/content confound is removed. (RSSM latent still carries a "
                     "reversed-history effect; WP/CP/embed are clean.)")
    else:
        mode_note = ("<b>Mode: physical navigation.</b> The backward run physically retraces "
                     "the path, so at a matched position the camera faces the opposite way — "
                     "post-alignment WP residual partly reflects this heading/content change, "
                     "not only structural inconsistency.")

    html = _TEMPLATE.format(
        ckpt_step=manifest["checkpoint_step"], n_ep=len(eps), n_pairs=len(rows),
        match_dist=match_max_dist, verdict="".join(verdict), mode=mode, mode_note=mode_note,
        ep_rows=ep_html, agg_rows=rows_html,
        curves=curves_b64, cloud=cloud_b64, rgb_f=rgb_f, rgb_b=rgb_b,
        scene=manifest["episodes"][0]["scene"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path


_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VGGT First-Frame Invariance Probe</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono&display=swap" rel="stylesheet">
<style>
  :root {{ --bg:#0e1116; --panel:#161b22; --fg:#e6edf3; --muted:#9aa7b4; --acc:#4cc9f0; --warn:#f6c177; }}
  body {{ margin:0; background:var(--bg); color:var(--fg); font-family:Inter,system-ui,sans-serif; line-height:1.6; }}
  .wrap {{ max-width:960px; margin:0 auto; padding:48px 24px 96px; }}
  h1 {{ font-size:2rem; font-weight:700; margin:0 0 .2em; }}
  h2 {{ font-size:1.3rem; margin-top:2.4em; border-bottom:1px solid #283039; padding-bottom:.3em; }}
  .sub {{ color:var(--muted); margin-top:0; }}
  code,.mono {{ font-family:"JetBrains Mono",monospace; font-size:.9em; }}
  .verdict {{ background:var(--panel); border-left:3px solid var(--acc); padding:18px 22px; border-radius:8px; margin:1.4em 0; }}
  table {{ border-collapse:collapse; width:100%; margin:1em 0; font-size:.9rem; }}
  th,td {{ text-align:left; padding:7px 10px; border-bottom:1px solid #283039; }}
  th {{ color:var(--muted); font-weight:600; }}
  td:not(:first-child) {{ font-family:"JetBrains Mono",monospace; }}
  img {{ width:100%; border-radius:8px; background:#fff; margin:.6em 0; }}
  .thumbs {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
  .thumbs figcaption {{ color:var(--muted); font-size:.85rem; text-align:center; }}
  .pill {{ display:inline-block; background:var(--panel); color:var(--muted); padding:2px 10px; border-radius:999px; font-size:.8rem; margin-right:6px; }}
</style></head>
<body><div class="wrap">
<h1>VGGT First-Frame Invariance Probe</h1>
<p class="sub">R2Dreamer L1 · checkpoint step {ckpt_step} · scene <span class="mono">{scene}</span> ·
mode <span class="mono">{mode}</span> · {n_ep} episode pairs ·
{n_pairs} position-matched frame pairs (≤ {match_dist} m)</p>

<div class="verdict">{verdict}</div>
<p class="sub">{mode_note}</p>

<h2>Setup</h2>
<p>VGGT emits World Points (WP, 37×37×3) and a Camera Pose (CP, 9-d) <b>relative to the first frame</b>
of its streaming window. The Dreamer encoder consumes the flattened 4116-d WP‖CP vector directly.
For each start pose we run a <b>forward</b> episode (start→goal) and a <b>backward</b> episode that
physically retraces the path (goal→start) with its own fresh VGGT window — so the same physical
viewpoint appears under two different first-frame references. We match frames by agent position and
compare the raw WP/CP, the WP after a rigid+scale <span class="mono">Umeyama</span> alignment, and the
trained RSSM posterior.</p>

<h2>Episodes</h2>
<table><tr><th>idx</th><th>episode_id</th><th>geodesic (m)</th><th>fwd steps</th><th>bwd steps</th><th>matched</th></tr>
{ep_rows}</table>

<h2>Same place, opposite heading</h2>
<p>A matched pair: identical agent position, but the backward run faces the other way and started its
VGGT window elsewhere.</p>
<div class="thumbs">
  <figure><img src="data:image/png;base64,{rgb_f}"><figcaption>forward frame</figcaption></figure>
  <figure><img src="data:image/png;base64,{rgb_b}"><figcaption>backward frame (matched position)</figcaption></figure>
</div>

<h2>Per-frame divergence</h2>
<img src="data:image/png;base64,{curves}">

<h2>WP point clouds — raw vs reference-frame-aligned</h2>
<p>If the backward cloud snaps onto the forward cloud after a single rigid+scale transform, the two
runs hold the <b>same geometry in different frames</b>.</p>
<img src="data:image/png;base64,{cloud}">

<h2>Aggregate metrics</h2>
<table><tr><th>metric</th><th>mean</th><th>std</th><th>min</th><th>max</th></tr>
{agg_rows}</table>
<p class="sub"><span class="pill">wp_residual_ratio</span> aligned RMSE ÷ raw RMSE — low ⇒ pure frame change.
<span class="pill">latent_js</span> JS divergence of RSSM posterior in nats (0 … ln2≈0.693).
<span class="pill">deter_cosine</span> cosine of the 2048-d deterministic state.</p>

</div></body></html>
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", default="output/analysis/invariance")
    ap.add_argument("--out", default="docs/3d-invariance-vggt-first-frame.html")
    ap.add_argument("--match-max-dist", type=float, default=0.30)
    a = ap.parse_args()
    p = build_report(REPO / a.artifact_dir if not Path(a.artifact_dir).is_absolute()
                     else a.artifact_dir,
                     REPO / a.out if not Path(a.out).is_absolute() else a.out,
                     a.match_max_dist)
    print("wrote", p)
