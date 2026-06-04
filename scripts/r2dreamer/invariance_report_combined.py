"""Merge the navigate + teleport invariance analyses into one HTML report.

White theme, simplified single-purpose figures, and an explainer of every number.
The verdict rests ONLY on what the VGGT pipeline actually emits (raw WP, CP, encoder
embedding) and the downstream Dreamer latents (belief JS, memory cosine). No rigid /
scale alignment is applied anywhere — alignment is not part of the pipeline, so it is
not part of the test.

Sections: what was done · what we found · how to read it · pro/against · takeaway.

    python scripts/r2dreamer/invariance_report_combined.py \
        --navigate-dir output/analysis/invariance \
        --teleport-dir output/analysis/invariance_teleport \
        --out docs/3d-invariance-vggt-first-frame.html
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.r2dreamer.invariance_report import (  # noqa: E402
    _fig_b64, _rgb_b64, load, compute)

LN2 = float(np.log(2))
NAV_C, TELE_C = "#e08a3c", "#2a9d8f"   # navigate / teleport accent colours

METRICS = [  # (key, label, unit) for the numeric table — raw pipeline outputs only
    ("wp_cosine", "Raw WP cosine", ""),
    ("wp_l2", "Raw WP L2", ""),
    ("embed_cosine", "Encoder embedding cosine", ""),
    ("cp_cosine", "Camera-pose cosine", ""),
    ("cp_l2", "Camera-pose L2", ""),
    ("latent_js", "RSSM belief JS", "nats"),
    ("deter_cosine", "RSSM memory cosine", ""),
]


def _mean(rows, k):
    return float(np.mean([r[k] for r in rows]))


def _chw_or_hwc(a):
    return np.transpose(a, (1, 2, 0)) if a.shape[0] == 3 else a


def _white_style(plt):
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "axes.edgecolor": "#cfcfcf",
        "axes.grid": True, "grid.color": "#ececec", "grid.linewidth": 1,
        "font.size": 12, "font.family": "DejaVu Sans",
        "axes.titlesize": 13.5, "axes.titleweight": "bold",
        "axes.labelcolor": "#222", "text.color": "#222",
        "xtick.color": "#444", "ytick.color": "#444",
        "axes.spines.top": False, "axes.spines.right": False,
    })


def build(nav_dir, tele_dir, out_path, match_max_dist=0.30):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    _white_style(plt)

    nav_m, nav_e = load(nav_dir)
    tele_m, tele_e = load(tele_dir)
    nav_rows, nav_match = compute(nav_e, match_max_dist)
    tele_rows, tele_match = compute(tele_e, match_max_dist)
    nav = {k: _mean(nav_rows, k) for k, _, _ in METRICS}
    tele = {k: _mean(tele_rows, k) for k, _, _ in METRICS}

    # =========================================================================
    # Fig A — ONE intuitive chart: similarity of the representation at the same
    # place, everything on a 0..1 "1 = identical" axis (teleport / clean mode).
    # Every bar is a RAW pipeline output or a downstream latent — no alignment.
    # =========================================================================
    sim_items = [
        ("WP — raw 3-D coords (model input)", tele["wp_cosine"]),
        ("Dreamer encoder embedding", tele["embed_cosine"]),
        ("Dreamer belief  (1 − JS/ln2)", 1 - tele["latent_js"] / LN2),
        ("Dreamer memory  (deter cosine)", tele["deter_cosine"]),
    ]
    labels = [a for a, _ in sim_items][::-1]
    vals = [b for _, b in sim_items][::-1]
    norm = mcolors.Normalize(0, 1)
    cmap = plt.get_cmap("RdYlGn")
    fig, ax = plt.subplots(figsize=(9.5, 3.9))
    y = np.arange(len(vals))
    ax.barh(y, vals, color=[cmap(norm(max(0, v))) for v in vals], edgecolor="#888", height=.6)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlim(-0.1, 1.08)
    ax.axvline(1.0, ls="--", color="#2a9d8f", lw=1.4)
    ax.axvline(0.0, ls="--", color="#c44", lw=1.2)
    ax.text(1.0, len(vals) - .35, "identical", color="#2a9d8f", ha="center", va="bottom", fontsize=9)
    ax.text(0.0, len(vals) - .35, "unrelated", color="#c44", ha="center", va="bottom", fontsize=9)
    for yi, v in zip(y, vals):
        ax.annotate(f"{v:.2f}", (max(v, 0) + .015, yi), va="center", fontsize=11, fontweight="bold")
    ax.set_title("Same place, two different starts — how similar is the representation?  (1 = identical)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); simbar = _fig_b64(fig)

    # =========================================================================
    # Fig B — raw WP point cloud at one matched place: forward vs reverse.
    # Exactly as the pipeline emits them (no alignment) — they sit in different
    # first-frame reference frames.
    # =========================================================================
    (p0, _) = tele_match[0]
    f, b = p0[len(p0) // 2]
    fwp = tele_e[0]["fwd_wp"][f].reshape(-1, 3)
    bwp = tele_e[0]["bwd_wp"][b].reshape(-1, 3)
    raw_cos = float(np.dot(fwp.ravel(), bwp.ravel()) /
                    (np.linalg.norm(fwp) * np.linalg.norm(bwp)))

    from matplotlib.ticker import MaxNLocator
    FWD_C, REV_C = "#2b6cb0", "#dd7a2b"

    def _wp_pixel_colors(rgb, grid=37):
        """Block-average each RGB image patch to the matching VGGT point."""
        img = _chw_or_hwc(np.asarray(rgb)).astype(np.float64)
        if img.max() > 1.5:
            img = img / 255.0
        s = (img.shape[0] // grid) * grid
        k = s // grid
        img = img[:s, :s].reshape(grid, k, grid, k, 3).mean((1, 3))
        return np.clip(img.reshape(-1, 3), 0, 1)

    cf = _wp_pixel_colors(tele_e[0]["fwd_rgb"][f])
    cb = _wp_pixel_colors(tele_e[0]["bwd_rgb"][b])
    cf_faint = np.column_stack([cf, np.full(len(cf), 0.76)])
    cb_faint = np.column_stack([cb, np.full(len(cb), 0.76)])

    # Convention: z = height (vertical), x-y = floor plane.
    # shared axis limits across the panels so the coordinate-frame shift stays visible
    both = np.vstack([fwp, bwp])
    lim = [(both[:, j].min(), both[:, j].max()) for j in (0, 1, 2)]  # x, y, z order
    spans = [max(hi - lo, 1e-6) for lo, hi in lim]

    def _clean3d(a):
        a.view_init(elev=32, azim=-118)
        a.set_xlim(lim[0]); a.set_ylim(lim[1]); a.set_zlim(lim[2])
        a.set_box_aspect(spans)
        for axis in (a.xaxis, a.yaxis, a.zaxis):  # white panes, sparse ticks
            axis.set_pane_color((1, 1, 1, 1)); axis.pane.set_edgecolor("#dcdcdc")
            axis.set_major_locator(MaxNLocator(3))
        a.tick_params(labelsize=11, pad=0)
        a.set_xlabel("x (m)", fontsize=11, labelpad=-2)
        a.set_ylabel("y (m)", fontsize=11, labelpad=-2)
        a.set_zlabel("height (m)", fontsize=11, labelpad=-5)
        a.grid(True)

    # Fig B — paper-facing diagnostic: the main evidence is the top-down overlay;
    # the oblique 3-D panel confirms the same raw clouds without forcing the
    # reader to mentally align two separate 3-D camera views.
    fig = plt.figure(figsize=(10.8, 5.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.16)

    ax_o = fig.add_subplot(gs[0, 0])
    ax_o.scatter(fwp[:, 0], fwp[:, 1], s=20, facecolors=cf_faint, edgecolors=FWD_C,
                 linewidths=.55, label="forward")
    ax_o.scatter(bwp[:, 0], bwp[:, 1], s=24, marker="^", facecolors=cb_faint,
                 edgecolors=REV_C, linewidths=.55, label="reverse")
    ax_o.set_aspect("equal")
    ax_o.set_xlabel("x (m)", fontsize=12); ax_o.set_ylabel("y (m)", fontsize=12)
    ax_o.tick_params(labelsize=11)
    ax_o.set_title("(a) floor-plane overlay", fontsize=12.5)
    ax_o.legend(fontsize=10, loc="lower center", ncol=2, frameon=True)

    ax_3d = fig.add_subplot(gs[0, 1], projection="3d")
    ax_3d.scatter(fwp[:, 0], fwp[:, 1], fwp[:, 2], s=12, c=cf, marker="o",
                  alpha=.66, depthshade=False, edgecolors=FWD_C, linewidths=.18)
    ax_3d.scatter(bwp[:, 0], bwp[:, 1], bwp[:, 2], s=13, c=cb, marker="^",
                  alpha=.68, depthshade=False, edgecolors=REV_C, linewidths=.18)
    _clean3d(ax_3d)
    ax_3d.set_title("(b) oblique 3-D view", fontsize=12.5)

    fig.suptitle("Same physical place, different VGGT first-frame coordinates",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.text(0.5, 0.018, "raw VGGT world-points, no rigid or scale alignment",
             ha="center", fontsize=10, color="#666")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.86, wspace=0.2)
    cloud_b64 = _fig_b64(fig)

    # Variant B — abstract run-colour points only: highest contrast, least texture.
    fig = plt.figure(figsize=(10.8, 5.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.16)
    ax_o = fig.add_subplot(gs[0, 0])
    ax_o.scatter(fwp[:, 0], fwp[:, 1], s=18, facecolors="none", edgecolors=FWD_C,
                 linewidths=.75, alpha=.72, label="forward")
    ax_o.scatter(bwp[:, 0], bwp[:, 1], s=22, marker="^", facecolors="none",
                 edgecolors=REV_C, linewidths=.75, alpha=.76, label="reverse")
    ax_o.set_aspect("equal")
    ax_o.set_xlabel("x (m)", fontsize=12); ax_o.set_ylabel("y (m)", fontsize=12)
    ax_o.tick_params(labelsize=11)
    ax_o.set_title("(a) floor-plane overlay", fontsize=12.5)
    ax_o.legend(fontsize=10, loc="lower center", ncol=2, frameon=True)
    ax_3d = fig.add_subplot(gs[0, 1], projection="3d")
    ax_3d.scatter(fwp[:, 0], fwp[:, 1], fwp[:, 2], s=12, c=FWD_C, marker="o",
                  alpha=.42, depthshade=False, edgecolors="none")
    ax_3d.scatter(bwp[:, 0], bwp[:, 1], bwp[:, 2], s=13, c=REV_C, marker="^",
                  alpha=.45, depthshade=False, edgecolors="none")
    _clean3d(ax_3d)
    ax_3d.set_title("(b) oblique 3-D view", fontsize=12.5)
    fig.suptitle("Variant B: abstract run-colour point cloud", fontsize=15, fontweight="bold", y=0.995)
    fig.text(0.5, 0.018, "maximal separation clarity, no image texture",
             ha="center", fontsize=10, color="#666")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.86, wspace=0.2)
    cloud_abstract_b64 = _fig_b64(fig)

    # Variant C — top-down density abstraction: reduces 1369 points per run to mass.
    fig, ax_d = plt.subplots(figsize=(7.3, 5.7))
    extent = [lim[0][0], lim[0][1], lim[1][0], lim[1][1]]
    ax_d.hist2d(fwp[:, 0], fwp[:, 1], bins=58, range=[[extent[0], extent[1]], [extent[2], extent[3]]],
                cmap="Blues", alpha=.72, cmin=1)
    ax_d.hist2d(bwp[:, 0], bwp[:, 1], bins=58, range=[[extent[0], extent[1]], [extent[2], extent[3]]],
                cmap="Oranges", alpha=.56, cmin=1)
    ax_d.set_aspect("equal")
    ax_d.set_xlabel("x (m)", fontsize=12); ax_d.set_ylabel("y (m)", fontsize=12)
    ax_d.set_title("Variant C: top-down occupancy density", fontsize=14, fontweight="bold")
    fig.tight_layout(); cloud_density_b64 = _fig_b64(fig)

    # Variant D — minimalist statistical abstraction: ellipses + sparse point sample.
    from matplotlib.patches import Ellipse

    def _cov_ellipse(points_xy, n_std, **kwargs):
        cov = np.cov(points_xy.T)
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 1e-9)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        width, height = 2 * n_std * np.sqrt(vals)
        return Ellipse(points_xy.mean(0), width, height, angle=angle, **kwargs)

    fxy = fwp[:, [0, 1]]; bxy = bwp[:, [0, 1]]
    fig, ax_m = plt.subplots(figsize=(7.2, 5.4))
    ax_m.scatter(fxy[::7, 0], fxy[::7, 1], s=10, c=FWD_C, alpha=.18)
    ax_m.scatter(bxy[::7, 0], bxy[::7, 1], s=12, c=REV_C, marker="^", alpha=.18)
    ax_m.add_patch(_cov_ellipse(fxy, 1.0, facecolor=FWD_C, edgecolor=FWD_C, alpha=.14, lw=2))
    ax_m.add_patch(_cov_ellipse(bxy, 1.0, facecolor=REV_C, edgecolor=REV_C, alpha=.14, lw=2))
    ax_m.add_patch(_cov_ellipse(fxy, 1.6, facecolor="none", edgecolor=FWD_C, alpha=.9, lw=2))
    ax_m.add_patch(_cov_ellipse(bxy, 1.6, facecolor="none", edgecolor=REV_C, alpha=.9, lw=2))
    ax_m.set_xlim(lim[0]); ax_m.set_ylim(lim[1]); ax_m.set_aspect("equal")
    ax_m.set_xlabel("x (m)", fontsize=12); ax_m.set_ylabel("y (m)", fontsize=12)
    ax_m.set_title("Variant D: minimal spread summary", fontsize=14, fontweight="bold")
    fig.tight_layout(); cloud_minimal_b64 = _fig_b64(fig)

    # Extra plot — top-down overlay: both clouds in ONE shared x-y floor frame
    # (looking down the height axis z), image-colour fills + run outlines.
    fig, ax_o = plt.subplots(figsize=(6.6, 6.0))
    ax_o.scatter(fwp[:, 0], fwp[:, 1], s=20, facecolors=cf_faint, edgecolors=FWD_C,
                 linewidths=.7, label="forward")
    ax_o.scatter(bwp[:, 0], bwp[:, 1], s=24, marker="^", facecolors=cb_faint,
                 edgecolors=REV_C, linewidths=.7, label="reverse")
    ax_o.set_aspect("equal")
    ax_o.set_xlabel("x (m)", fontsize=12); ax_o.set_ylabel("y (m)", fontsize=12)
    ax_o.tick_params(labelsize=11)
    ax_o.set_title("Top-down (floor plane x–y) — both runs, one shared frame",
                   fontsize=12.5)
    ax_o.legend(fontsize=10, loc="best")
    fig.tight_layout(); topdown_b64 = _fig_b64(fig)

    # =========================================================================
    # Fig C — is the belief gap consistent along the path? (per-frame JS).
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    all_curves = []
    maxlen = 0
    for i in sorted({r["episode"] for r in tele_rows}):
        s = sorted([r for r in tele_rows if r["episode"] == i], key=lambda r: r["f"])
        xs = [r["f"] for r in s]; ys = [r["latent_js"] for r in s]
        ax.plot(xs, ys, color="#bcd", lw=1)
        all_curves.append(ys); maxlen = max(maxlen, len(ys))
    padded = np.full((len(all_curves), maxlen), np.nan)
    for r, c in enumerate(all_curves):
        padded[r, :len(c)] = c
    mean = np.nanmean(padded, 0)
    ax.plot(range(maxlen), mean, color=TELE_C, lw=2.6, label="mean over episodes")
    ax.axhline(LN2, ls="--", color="#c44", lw=1.2)
    ax.text(maxlen - 1, LN2, " max (ln2)", color="#c44", va="center", fontsize=9)
    ax.set_ylim(0, 0.72); ax.set_xlabel("frame along the path"); ax.set_ylabel("belief JS (nats)")
    ax.set_title("The belief gap is large and steady the whole way (teleport)")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout(); js_b64 = _fig_b64(fig)

    # ---- matched RGB thumbnails (both modes) ------------------------------
    (npr, _) = nav_match[0]; nf, nb = npr[len(npr) // 2]
    nav_f = _rgb_b64(_chw_or_hwc(nav_e[0]["fwd_rgb"][nf]))
    nav_b = _rgb_b64(_chw_or_hwc(nav_e[0]["bwd_rgb"][nb]))
    tele_f = _rgb_b64(_chw_or_hwc(tele_e[0]["fwd_rgb"][f]))
    tele_b = _rgb_b64(_chw_or_hwc(tele_e[0]["bwd_rgb"][b]))

    # ---- numeric table -----------------------------------------------------
    def trow(k, lab, unit):
        u = f" {unit}" if unit else ""
        return (f"<tr><td>{lab}</td><td>{nav[k]:.3f}{u}</td><td>{tele[k]:.3f}{u}</td></tr>")
    metric_rows = "".join(trow(k, lab, u) for k, lab, u in METRICS)

    html = _TEMPLATE.format(
        ckpt_step=nav_m["checkpoint_step"], scene=nav_m["episodes"][0]["scene"],
        n_nav=len(nav_rows), n_tele=len(tele_rows),
        simbar=simbar, cloud=cloud_b64, topdown=topdown_b64, js=js_b64,
        cloud_abstract=cloud_abstract_b64, cloud_density=cloud_density_b64,
        cloud_minimal=cloud_minimal_b64,
        nav_f=nav_f, nav_b=nav_b, tele_f=tele_f, tele_b=tele_b, metric_rows=metric_rows,
        wp_cos=f"{tele['wp_cosine']:.2f}", embed_cos=f"{tele['embed_cosine']:.2f}",
        js_tele=f"{tele['latent_js']:.2f}", js_pct=f"{tele['latent_js'] / LN2 * 100:.0f}",
        deter=f"{tele['deter_cosine']:.2f}", js_nav=f"{nav['latent_js']:.2f}",
        raw_cos=f"{raw_cos:.2f}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path


_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VGGT First-Frame Invariance — Navigate &amp; Teleport</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono&display=swap" rel="stylesheet">
<style>
  :root {{ --bg:#ffffff; --panel:#f6f8fa; --fg:#1c2128; --muted:#5b6570; --acc:#2a7de1;
           --pro:#1a8f5c; --con:#d97a2b; --line:#e4e8ec; }}
  body {{ margin:0; background:var(--bg); color:var(--fg); font-family:Inter,system-ui,sans-serif; line-height:1.65; }}
  .wrap {{ max-width:960px; margin:0 auto; padding:48px 24px 110px; }}
  h1 {{ font-size:2rem; font-weight:700; margin:0 0 .15em; }}
  h2 {{ font-size:1.3rem; margin-top:2.6em; border-bottom:2px solid var(--line); padding-bottom:.3em; }}
  h3 {{ font-size:1.04rem; margin:1.4em 0 .3em; }}
  .sub {{ color:var(--muted); margin-top:0; }}
  code,.mono {{ font-family:"JetBrains Mono",monospace; font-size:.9em; }}
  .verdict {{ background:var(--panel); border-left:4px solid var(--acc); padding:18px 22px; border-radius:8px; margin:1.4em 0; }}
  figure {{ margin:1.2em 0; }}
  figcaption {{ color:var(--muted); font-size:.9rem; margin-top:.3em; }}
  table {{ border-collapse:collapse; width:100%; margin:1em 0; font-size:.92rem; }}
  th,td {{ text-align:left; padding:8px 12px; border-bottom:1px solid var(--line); }}
  th {{ color:var(--muted); font-weight:600; }}
  td:not(:first-child) {{ font-family:"JetBrains Mono",monospace; }}
  tr:hover td {{ background:#fafcff; }}
  img {{ width:100%; border:1px solid var(--line); border-radius:8px; }}
  .thumbs {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; }}
  .thumbs img {{ border-radius:6px; }}
  .thumbs figcaption {{ font-size:.78rem; text-align:center; }}
  .read {{ background:#f0f6ff; border:1px solid #d7e6fb; border-radius:8px; padding:6px 22px; margin:1.2em 0; }}
  .read li {{ margin:.5em 0; }}
  .note {{ background:#fff8ec; border:1px solid #f3e2c0; border-radius:8px; padding:6px 22px; margin:1.2em 0; }}
  .cols {{ display:grid; grid-template-columns:1fr 1fr; gap:22px; margin-top:1em; }}
  .card {{ background:var(--panel); border-radius:10px; padding:18px 20px; }}
  .card.pro {{ border-top:4px solid var(--pro); }}
  .card.con {{ border-top:4px solid var(--con); }}
  .card h3 {{ margin-top:0; }} .card.pro h3 {{ color:var(--pro); }} .card.con h3 {{ color:var(--con); }}
  .card li {{ margin-bottom:.6em; }}
  .pill {{ display:inline-block; background:#eaf2fe; color:var(--acc); padding:1px 9px; border-radius:999px; font-size:.78rem; }}
  @media(max-width:760px){{ .cols,.thumbs{{grid-template-columns:1fr 1fr;}} }}
</style></head>
<body><div class="wrap">

<h1>VGGT First-Frame Invariance Probe</h1>
<p class="sub">R2Dreamer L1 · 2M-step checkpoint (step {ckpt_step}) · HM3D house
<span class="mono">{scene}</span> · two protocols: <b>navigate</b> ({n_nav} matched
frame pairs) and <b>teleport</b> ({n_tele} matched frame pairs)</p>

<div class="verdict">
At the <b>same physical viewpoint</b> reached from two different starts — in <b>teleport</b> the very same
camera pose, so the RGB input image is identical — the raw VGGT World-Points the world model ingests are
essentially <b>unrelated</b> (cosine {wp_cos}) and the encoder embedding cosine is only {embed_cos}. The only
thing that changed is <b>which frame VGGT took as its first-frame reference</b>. It propagates downstream: the
trained model's belief diverges by <b>JS {js_tele} nats</b> ({js_pct}% of maximum; memory cosine {deter}),
identically under both protocols. <b>The 2M-step world model is not invariant to the start pose — the
first-frame-relative encoding flows straight into its latent state.</b>
</div>

<div class="note"><b>Method note.</b> Every number and figure below is computed <i>directly from what the VGGT
pipeline emits</i> (raw World-Points, Camera-Pose, the encoder embedding) and from the Dreamer RSSM latents.
<b>No rigid or scale alignment is applied anywhere</b> — alignment is not part of the pipeline, so it is not
part of the test. We compare the representations exactly as the model produces and consumes them.</div>

<h2>1 · What we did</h2>
<p>VGGT turns each RGB frame into World Points (WP, 37×37×3) and a Camera Pose (CP, 9-d) expressed
<b>relative to the first frame</b> of its streaming window; the Dreamer encoder consumes the flattened
4116-d WP‖CP vector. If the encoding were invariant, the same physical state would map to the same
representation regardless of where the episode started. We test this with two protocols (5 start poses in
one house, ≈30 frames each), each comparing a <b>forward</b> pass with a <b>backward</b> pass that revisits
the same places in a fresh VGGT window:</p>
<ul>
<li><span class="pill">navigate</span> the agent <b>physically</b> walks back — same positions, but the
camera faces the opposite way (realistic; heading also changes, so the RGB differs too).</li>
<li><span class="pill">teleport</span> the agent is placed at the <b>exact</b> forward poses (same position
<i>and</i> heading) in reverse, so the <b>RGB image is identical</b>. Only the first-frame reference differs
— the clean test.</li>
</ul>
<div class="thumbs">
  <figure><img src="data:image/png;base64,{nav_f}"><figcaption>navigate · forward</figcaption></figure>
  <figure><img src="data:image/png;base64,{nav_b}"><figcaption>navigate · backward (opposite heading)</figcaption></figure>
  <figure><img src="data:image/png;base64,{tele_f}"><figcaption>teleport · forward</figcaption></figure>
  <figure><img src="data:image/png;base64,{tele_b}"><figcaption>teleport · replay (identical image)</figcaption></figure>
</div>

<h2>2 · What we found</h2>
<figure><img src="data:image/png;base64,{simbar}">
<figcaption><b>The one chart to read.</b> Each bar is how similar the two runs' representation of the
<i>same physical place</i> is, on a 0…1 scale where 1 = identical. The raw coordinates fed to the model are
near 0, the encoder embedding only {embed_cos}, and the model's own belief/memory stay well below 1 — none of
them is invariant to where the run started.</figcaption></figure>

<figure><img src="data:image/png;base64,{cloud}">
<figcaption><b>Variant A — real-colour hybrid.</b> The paper-facing view overlays
both runs in the same axes: each point is filled with its real camera-pixel colour, with blue circle
outlines for forward and orange triangle outlines for reverse. The floor-plane panel shows the main effect
directly, while the oblique 3-D panel confirms the same raw clouds with z as height. No alignment is applied;
this is the vector the Dreamer encoder receives. Whole-vector cosine between the two clouds at this step is
{raw_cos}.</figcaption></figure>

<figure><img src="data:image/png;base64,{cloud_abstract}">
<figcaption><b>Variant B — abstract run-colour point cloud.</b> This removes the RGB texture and uses only
blue/orange marker identity. It is less faithful to the camera image, but the coordinate-frame separation reads
fastest.</figcaption></figure>

<figure><img src="data:image/png;base64,{cloud_density}">
<figcaption><b>Variant C — top-down occupancy density.</b> This compresses the dense point cloud into a
floor-plane density map. It is useful if the individual 37×37 point lattice feels too busy for the main
paper figure.</figcaption></figure>

<figure><img src="data:image/png;base64,{cloud_minimal}">
<figcaption><b>Variant D — spread summary.</b> This is the most schematic option: a small point sample and
covariance ellipses. It is clean, but abstracts away the raw lattice structure.</figcaption></figure>

<figure><img src="data:image/png;base64,{topdown}">
<figcaption><b>Top-down zoom, one shared frame.</b> The same two clouds projected onto the floor plane (x–y,
looking down the height axis z) and overlaid in a single coordinate frame: real image-colour fills, forward
as blue-outlined ○, reverse as orange-outlined △. The clouds describe the same room but live in two different
first-frame coordinate systems.</figcaption></figure>

<figure><img src="data:image/png;base64,{js}">
<figcaption><b>The model's belief gap is persistent.</b> Jensen–Shannon divergence between the two RSSM
posteriors stays near 0.4–0.5 nats along the whole path (max possible = ln2 ≈ 0.69) — not a transient.</figcaption></figure>

<h2>3 · How to read the numbers</h2>
<div class="read"><ul>
<li><b>Cosine</b> (raw WP, embedding, memory): 1 = identical direction, 0 = unrelated, &lt;0 = opposed.
Raw 3-D coordinates of the same room in two first-frames are rotated/translated, so their cosine sits near
0 even though the scene is identical — and this is the actual vector the model ingests.</li>
<li><b>Why embedding cosine ({embed_cos}) &gt; raw WP cosine ({wp_cos})?</b> The encoder is a single learned
linear map <span class="mono">W·x + b</span>. The bias contributes little; the weight <span class="mono">W</span>
projects onto directions that <i>partly</i> overlap across frames — so it leaks a bit less, but a linear
layer cannot undo a start-dependent rotation, so it stays far below 1.</li>
<li><b>Belief JS (nats):</b> distance between the two posterior belief distributions. 0 = same belief,
ln2 ≈ 0.69 = no overlap. Our {js_tele} ≈ {js_pct}% of max ⇒ the model thinks it is in a substantially
different situation at the same spot.</li>
<li><b>Memory cosine (deter):</b> similarity of the 2048-d recurrent state. {deter} ⇒ partially aligned,
far from identical.</li>
<li><b>Navigate vs teleport.</b> In navigate the camera also faces the other way, so the RGB <i>and</i> the
frame differ; in teleport the RGB is identical and only the first-frame reference differs. The belief JS is
essentially the same in both ({js_nav} → {js_tele}) — so the non-invariance is not an artefact of the
heading change, it is the first-frame reference itself.</li>
</ul></div>

<table><tr><th>metric</th><th>navigate</th><th>teleport (clean)</th></tr>{metric_rows}</table>

<h2>4 · Is this actually a problem? — pro &amp; against</h2>
<div class="cols">
  <div class="card pro"><h3>▲ Pro — a real problem</h3><ul>
    <li><b>No stable world anchor in the input.</b> The same state maps to WP with cosine ≈ 0 depending on
        the arbitrary start frame — even when the RGB image is byte-for-byte identical (teleport).</li>
    <li><b>It reaches the belief state.</b> JS ≈ {js_tele} nats ({js_pct}% of max), memory cosine {deter}:
        the latent is not a function of physical state alone — against the world-model premise.</li>
    <li><b>Robust, not an artefact.</b> Essentially identical under navigate and teleport, so it is not the
        heading confound.</li>
    <li><b>Likely hurts cross-start generalization</b> and any representation reuse across episodes.</li>
  </ul></div>
  <div class="card con"><h3>▼ Against — may not matter / fixable</h3><ul>
    <li><b>Same scene, moving origin.</b> The WP are not corrupted, only re-expressed relative to a different
        first frame — the geometric information is present, so an expressive model <i>could</i> in principle
        learn a frame-invariant feature.</li>
    <li><b>Within an episode the reference is fixed,</b> so the representation is perfectly stable for
        within-episode control — all ObjectNav deployment needs.</li>
    <li><b>The task only needs relative geometry</b> ("goal relative to me now"), which is preserved.</li>
    <li><b>Our belief metric is cross-episode;</b> the model never matches latents across episodes at
        deploy, so the gap may overstate task-relevant harm.</li>
    <li><b>Magnitude is moderate</b> — memory cosine {deter} is well above unrelated.</li>
  </ul></div>
</div>

<h2>5 · Takeaway</h2>
<p>The first-frame-relative property of VGGT is real and propagates into the world model's latent — a
concern for <i>cross-start generalization and representation reuse</i>, though it is an information-preserving
frame ambiguity (the same room re-expressed in a moving origin) and is benign for within-episode control.
Open question for the thesis: does anchoring WP/CP to a <b>world / gravity-aligned frame</b> (or a learned
canonicalization) <i>inside the pipeline</i> improve transfer across start poses? The metrics here — raw-WP
cosine, encoder-embedding cosine, belief JS — are the yardstick to test it.</p>

</div></body></html>
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--navigate-dir", default="output/analysis/invariance")
    ap.add_argument("--teleport-dir", default="output/analysis/invariance_teleport")
    ap.add_argument("--out", default="docs/3d-invariance-vggt-first-frame.html")
    ap.add_argument("--match-max-dist", type=float, default=0.30)
    a = ap.parse_args()
    rp = lambda p: (REPO / p) if not Path(p).is_absolute() else Path(p)  # noqa: E731
    out = build(rp(a.navigate_dir), rp(a.teleport_dir), rp(a.out), a.match_max_dist)
    print("wrote", out)
