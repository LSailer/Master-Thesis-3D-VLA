"""vggt_arch.py — Generate a 4-panel TikZ architecture diagram for VGGT-1B.

Usage:
    uv run python visualizations/vggt_arch.py   # writes + compiles media/images/vggt_arch.pdf

All shape annotations come from live forward-hook captures via vggt_shapes.extract_shapes().
"""

import sys, os, subprocess, glob, textwrap

# ── Path setup ────────────────────────────────────────────────────────────────

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VGGT_PATH = os.path.join(REPO_ROOT, "external", "VGGT")
if VGGT_PATH not in sys.path:
    sys.path.insert(0, VGGT_PATH)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt(shape, squeeze_batch=True) -> str:
    """Format a shape tuple as a compact string like [1,5,930,2048]."""
    if isinstance(shape, int):
        return str(shape)
    if squeeze_batch and len(shape) >= 2 and shape[0] == 1:
        inner = ",".join(str(d) for d in shape[1:])
        return f"[{inner}]"
    return "[" + ",".join(str(d) for d in shape) + "]"


def _bs(shape) -> str:
    """Return B·S part label like '(B·S=5)'."""
    if isinstance(shape, tuple) and len(shape) >= 1:
        return str(shape[0])
    return "?"


def generate_tikz(shapes: dict) -> str:
    """Build the complete LaTeX/TikZ source string from the shapes dict."""

    # Convenience accessors with safe fallbacks
    def s(key, default=()):
        return shapes.get(key, default)

    inp            = s("input",            (1, 3, 3, 350, 518))
    patch_emb      = s("patch_embed",      (3, 925, 1024))
    tok_cat        = s("tokens_concat",    (3, 930, 1024))
    fb_first       = s("frame_block_first",(3, 930, 1024))
    gb_first       = s("global_block_first",(1, 2790, 1024))
    aggregated     = s("aggregated",       (1, 3, 930, 2048))
    psi            = shapes.get("patch_start_idx", 5)
    dp0            = s("dpt_project_0",    (3, 256, 25, 37))
    dp1            = s("dpt_project_1",    (3, 512, 25, 37))
    dp2            = s("dpt_project_2",    (3,1024, 25, 37))
    dp3            = s("dpt_project_3",    (3,1024, 25, 37))
    dr0            = s("dpt_resize_0",     (3, 256,100,148))
    dr1            = s("dpt_resize_1",     (3, 512, 50, 74))
    dr2            = s("dpt_resize_2",     (3,1024, 25, 37))
    dr3            = s("dpt_resize_3",     (3,1024, 13, 19))
    drf4           = s("dpt_refinenet4",   (3, 256, 25, 37))
    drf1           = s("dpt_refinenet1",   (3, 256,200,296))
    dout           = s("dpt_output",       (3,   2,350, 518))
    cam_norm       = s("camera_token_norm",(1, 3, 2048))
    cam_trunk      = s("camera_trunk_last",(1, 3, 2048))
    cam_pose       = s("camera_pose_branch",(1,3, 9))
    trk_feat       = s("track_features",  (1, 3, 128, 175, 259))

    B  = inp[0] if len(inp) > 0 else 1
    S  = inp[1] if len(inp) > 1 else 3
    H  = inp[3] if len(inp) > 3 else 350
    W  = inp[4] if len(inp) > 4 else 518

    # Helper: compact shape string for annotations
    def sh(t, skip_first=False):
        if not t:
            return "?"
        if skip_first:
            t = t[1:]
        return "[" + ",".join(str(d) for d in t) + "]"

    # Build per-panel TeX strings
    # ── Panel A: Overview ────────────────────────────────────────────────────
    panelA = rf"""
  %% ── Panel A: Overview ──────────────────────────────────────────────────
  \node[block,  minimum width=1.8cm] (input)  at (0,0)  {{Input}};
  \node[shapelabel, below=1pt of input] (input_s) {{\scriptsize{sh(inp)}}};

  \node[block,  minimum width=2.2cm, right=1.2cm of input]  (agg)   {{Aggregator}};
  \node[shapelabel, below=1pt of agg]   (agg_s)  {{\scriptsize{sh(aggregated)}}};

  \node[block,  minimum width=2.0cm, right=1.4cm of agg, yshift= 1.6cm]  (camH)  {{CameraHead}};
  \node[block,  minimum width=2.0cm, right=1.4cm of agg, yshift= 0.4cm]  (dptD)  {{DPTHead (depth)}};
  \node[block,  minimum width=2.0cm, right=1.4cm of agg, yshift=-0.8cm]  (dptP)  {{DPTHead (point)}};
  \node[optblock, minimum width=2.0cm, right=1.4cm of agg, yshift=-2.0cm] (trkH)  {{TrackHead*}};

  \node[shapelabel, right=0.6cm of camH] (camH_s) {{\scriptsize{sh(cam_pose)}}};
  \node[shapelabel, right=0.6cm of dptD] (dptD_s) {{\scriptsize[B,S,{H},{W},1]}};
  \node[shapelabel, right=0.6cm of dptP] (dptP_s) {{\scriptsize[B,S,{H},{W},3]}};
  \node[shapelabel, right=0.6cm of trkH] (trkH_s) {{\scriptsize[B,S,N,2]}};

  \draw[->, thick] (input) -- (agg);
  \draw[->, thick] (agg.east) -- ++(0.5,0) |- (camH.west);
  \draw[->, thick] (agg.east) -- ++(0.5,0) |- (dptD.west);
  \draw[->, thick] (agg.east) -- ++(0.5,0) |- (dptP.west);
  \draw[->, thick, dashed] (agg.east) -- ++(0.5,0) |- (trkH.west);
  \draw[->, thick] (camH.east) -- (camH_s);
  \draw[->, thick] (dptD.east) -- (dptD_s);
  \draw[->, thick] (dptP.east) -- (dptP_s);
  \draw[->, thick, dashed] (trkH.east) -- (trkH_s);

  \node[anchor=north west, font=\bfseries\small] at (-0.9,0.8) {{A\quad Overview}};
  \node[shapelabel, below=2pt of trkH] {{\scriptsize* optional: needs query\_points}};
"""

    # ── Panel B: Aggregator ──────────────────────────────────────────────────
    panelB = rf"""
  %% ── Panel B: Aggregator ────────────────────────────────────────────────
  \node[block, minimum width=2.2cm] (pe)  at (0,-5.5) {{PatchEmbed}};
  \node[shapelabel, below=1pt of pe]  {{\scriptsize{sh(patch_emb)}}};

  \node[block, fill=yellow!20, minimum width=0.7cm, above right=0.1cm and 0.3cm of pe] (camtok) {{cam}};
  \node[block, fill=gray!20,   minimum width=1.0cm, right=0.05cm of camtok]             (regtok) {{reg×{psi-1}}};

  \node[block, minimum width=2.2cm, right=1.0cm of pe, yshift=-0.1cm] (cat) {{cat → {sh(tok_cat)}}};

  \node[block, minimum width=3.0cm, below=1.2cm of cat, dashed] (loop)
      {{\parbox{{2.8cm}}{{\centering×24\\[2pt]Frame-Attn Block\\Global-Attn Block}}}};
  \node[shapelabel, below=1pt of loop] {{\scriptsize frame: {sh(fb_first)}}};
  \node[shapelabel, below=12pt of loop] {{\scriptsize global: {sh(gb_first)}}};

  \node[block, minimum width=2.2cm, below=1.6cm of loop] (agg_out)
      {{concat 1024+1024→2048}};
  \node[shapelabel, below=1pt of agg_out] {{\scriptsize{sh(aggregated)}}};

  \draw[->, thick] (pe)     -- (cat);
  \draw[->, thick] (camtok) |- (cat);
  \draw[->, thick] (regtok) |- (cat);
  \draw[->, thick] (cat)    -- (loop);
  \draw[->, thick] (loop)   -- (agg_out);

  \node[anchor=north west, font=\bfseries\small] at (-0.9,-4.7) {{B\quad Aggregator}};
"""

    # ── Panel C: DPT Head ────────────────────────────────────────────────────
    # 4 rows, each: project → resize; then FPN column
    panelC = rf"""
  %% ── Panel C: DPT Head ──────────────────────────────────────────────────
  \node[block, minimum width=1.6cm] (proj0) at (9.0,-4.9) {{proj}};
  \node[block, minimum width=1.6cm, right=0.6cm of proj0] (rsz0)  {{resize×4}};
  \node[shapelabel, below=1pt of proj0] {{\scriptsize{sh(dp0)}}};
  \node[shapelabel, below=1pt of rsz0]  {{\scriptsize{sh(dr0)}}};
  \node[shapelabel, left=0.2cm of proj0, font=\scriptsize\ttfamily] {{L4}};

  \node[block, minimum width=1.6cm, below=0.7cm of proj0] (proj1) {{proj}};
  \node[block, minimum width=1.6cm, right=0.6cm of proj1] (rsz1)  {{resize×2}};
  \node[shapelabel, below=1pt of proj1] {{\scriptsize{sh(dp1)}}};
  \node[shapelabel, below=1pt of rsz1]  {{\scriptsize{sh(dr1)}}};
  \node[shapelabel, left=0.2cm of proj1, font=\scriptsize\ttfamily] {{L11}};

  \node[block, minimum width=1.6cm, below=0.7cm of proj1] (proj2) {{proj}};
  \node[block, minimum width=1.6cm, right=0.6cm of proj2] (rsz2)  {{identity}};
  \node[shapelabel, below=1pt of proj2] {{\scriptsize{sh(dp2)}}};
  \node[shapelabel, below=1pt of rsz2]  {{\scriptsize{sh(dr2)}}};
  \node[shapelabel, left=0.2cm of proj2, font=\scriptsize\ttfamily] {{L17}};

  \node[block, minimum width=1.6cm, below=0.7cm of proj2] (proj3) {{proj}};
  \node[block, minimum width=1.6cm, right=0.6cm of proj3] (rsz3)  {{conv↓2}};
  \node[shapelabel, below=1pt of proj3] {{\scriptsize{sh(dp3)}}};
  \node[shapelabel, below=1pt of rsz3]  {{\scriptsize{sh(dr3)}}};
  \node[shapelabel, left=0.2cm of proj3, font=\scriptsize\ttfamily] {{L23}};

  %% FPN column
  \node[block, fill=green!10, minimum width=1.6cm, right=0.6cm of rsz3] (rn4) {{refinenet4}};
  \node[block, fill=green!10, minimum width=1.6cm, above=0.4cm of rn4]  (rn3) {{refinenet3}};
  \node[block, fill=green!10, minimum width=1.6cm, above=0.4cm of rn3]  (rn2) {{refinenet2}};
  \node[block, fill=green!10, minimum width=1.6cm, above=0.4cm of rn2]  (rn1) {{refinenet1}};
  \node[shapelabel, right=0.2cm of rn4] {{\scriptsize{sh(drf4)}}};
  \node[shapelabel, right=0.2cm of rn1] {{\scriptsize{sh(drf1)}}};

  \node[block, minimum width=1.6cm, above=0.4cm of rn1]  (conv2) {{output\_conv2}};
  \node[shapelabel, right=0.2cm of conv2] {{\scriptsize{sh(dout)}}};

  \draw[->, thick] (proj0) -- (rsz0);
  \draw[->, thick] (proj1) -- (rsz1);
  \draw[->, thick] (proj2) -- (rsz2);
  \draw[->, thick] (proj3) -- (rsz3);
  \draw[->, thick] (rsz3.east) -- (rn4.west);
  \draw[->, thick] (rsz2.east) -- (rn3.west);
  \draw[->, thick] (rsz1.east) -- (rn2.west);
  \draw[->, thick] (rsz0.east) -- (rn1.west);
  \draw[->, thick] (rn4) -- (rn3);
  \draw[->, thick] (rn3) -- (rn2);
  \draw[->, thick] (rn2) -- (rn1);
  \draw[->, thick] (rn1) -- (conv2);

  \node[anchor=north west, font=\bfseries\small] at (8.1,-4.2) {{C\quad DPT Head}};
"""

    # ── Panel D: CameraHead ──────────────────────────────────────────────────
    panelD = rf"""
  %% ── Panel D: CameraHead ────────────────────────────────────────────────
  \node[block, minimum width=2.0cm] (ctn) at (16.0,-4.9) {{token\_norm}};
  \node[shapelabel, below=1pt of ctn] {{\scriptsize{sh(cam_norm)}}};

  \node[block, minimum width=2.0cm, dashed, below=0.8cm of ctn] (loop2)
      {{\parbox{{1.8cm}}{{\centering×4 iters\\[3pt]embed\_pose\\AdaLN mod\\trunk[4 blocks]\\pose\_branch Δ}}}};
  \node[shapelabel, right=0.2cm of loop2, yshift=0.5cm] {{\scriptsize{sh(cam_trunk)}}};

  \node[block, minimum width=2.0cm, below=1.5cm of loop2] (cpose) {{pose\_enc}};
  \node[shapelabel, below=1pt of cpose] {{\scriptsize{sh(cam_pose)}}};

  \draw[->, thick] (ctn)   -- (loop2);
  \draw[->, thick] (loop2) -- (cpose);

  \node[anchor=north west, font=\bfseries\small] at (15.1,-4.2) {{D\quad CameraHead}};
"""

    # ── Assemble full LaTeX document ─────────────────────────────────────────
    doc = rf"""\documentclass[tikz,border=8pt]{{standalone}}
\usepackage{{tikz}}
\usetikzlibrary{{shapes.geometric,arrows.meta,fit,positioning,backgrounds,calc}}

\tikzstyle{{block}} = [rectangle, rounded corners=3pt, draw=black!60,
    fill=blue!8, minimum height=0.7cm, text centered, font=\small,
    inner sep=4pt]
\tikzstyle{{optblock}} = [block, dashed, fill=gray!8]
\tikzstyle{{shapelabel}} = [font=\scriptsize\ttfamily, text=black!55,
    inner sep=1pt, text centered]

\begin{{document}}
\begin{{tikzpicture}}[node distance=0.5cm and 0.7cm,
                     every path/.style={{>=Stealth}}]

{panelA}
{panelB}
{panelC}
{panelD}

  %% Divider lines between panels
  \draw[gray!40, thick, dashed] (-1,-4.2) -- (20,-4.2);

\end{{tikzpicture}}
\end{{document}}
"""
    return doc


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import glob

    device = (
        __import__("torch").device("mps")  if __import__("torch").backends.mps.is_available()  else
        __import__("torch").device("cuda") if __import__("torch").cuda.is_available()           else
        __import__("torch").device("cpu")
    )
    import torch

    print(f"Device: {device}")
    print("Loading VGGT-1B …")

    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vggt_shapes import extract_shapes

    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model = model.to(device).eval()

    img_dir = os.path.join(VGGT_PATH, "examples", "kitchen", "images")
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))[:3]
    print(f"Using {len(img_paths)} kitchen frames")

    images = load_and_preprocess_images(img_paths, mode="crop")
    images_dev = images.to(device)

    print("Extracting shapes …")
    shapes = extract_shapes(model, images_dev)

    print(f"\n{'Key':<28} {'Shape / Value':>30}")
    print("-" * 60)
    for k, v in sorted(shapes.items()):
        print(f"{k:<28} {str(v):>30}")

    print("\nGenerating TikZ …")
    tex_src = generate_tikz(shapes)

    out_dir = os.path.join(REPO_ROOT, "media", "images")
    out_tex = os.path.join(out_dir, "vggt_arch.tex")
    with open(out_tex, "w") as f:
        f.write(tex_src)
    print(f"Wrote: {out_tex}")

    # Compile with pdflatex (mactex installed at /Library/TeX/texbin/pdflatex)
    pdflatex = "/Library/TeX/texbin/pdflatex"
    try:
        result = subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-output-directory", out_dir, out_tex],
            capture_output=True, text=True, timeout=60
        )
        pdf_path = out_tex.replace(".tex", ".pdf")
        if os.path.exists(pdf_path):
            print(f"PDF compiled: {pdf_path}")
        else:
            print("pdflatex error — check media/images/vggt_arch.log")
    except FileNotFoundError:
        print("pdflatex not found — skipping PDF compilation")
    except subprocess.TimeoutExpired:
        print("pdflatex timed out")


if __name__ == "__main__":
    main()
