"""vggt_manim.py — Manim animation of the VGGT forward pass.

Render command:
    uv run manim -qm visualizations/vggt_manim.py VGGTForwardPass

The scene calls extract_shapes() at startup to get real tensor dimensions from
the live model, then uses those shapes in every Text label.
"""

from __future__ import annotations

import sys, os

# ── Path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VGGT_PATH = os.path.join(REPO_ROOT, "external", "VGGT")
for p in (VGGT_PATH, REPO_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from manim import *


# ── Helpers ───────────────────────────────────────────────────────────────────

BLUE_BOX   = {"fill_color": BLUE_E,   "fill_opacity": 0.25, "stroke_color": BLUE_D,   "stroke_width": 2}
GREEN_BOX  = {"fill_color": GREEN_E,  "fill_opacity": 0.25, "stroke_color": GREEN_D,  "stroke_width": 2}
GOLD_BOX   = {"fill_color": GOLD,     "fill_opacity": 0.35, "stroke_color": GOLD_E,   "stroke_width": 2}
GRAY_BOX   = {"fill_color": GRAY,     "fill_opacity": 0.20, "stroke_color": GRAY_E,   "stroke_width": 2}
ORANGE_BOX = {"fill_color": ORANGE,   "fill_opacity": 0.25, "stroke_color": "#c05000", "stroke_width": 2}
DASHED_BOX = {"fill_color": GRAY_B,   "fill_opacity": 0.15, "stroke_color": GRAY,     "stroke_width": 1.5}

SHAPE_COLOR = "#888888"


def box(w: float, h: float, style: dict, label: str = "", font_size: int = 22):
    """Return a RoundedRectangle + optional label as a VGroup."""
    r = RoundedRectangle(width=w, height=h, corner_radius=0.12, **style)
    if label:
        t = Text(label, font_size=font_size).move_to(r.get_center())
        return VGroup(r, t)
    return VGroup(r)


def shape_label(shape, font_size: int = 16) -> Text:
    if isinstance(shape, tuple):
        txt = "[" + ",".join(str(d) for d in shape) + "]"
    else:
        txt = str(shape)
    return Text(txt, font_size=font_size, color=SHAPE_COLOR)


def _load_shapes() -> dict:
    """Load VGGT and run extract_shapes(). Returns dict on success, {} on failure."""
    try:
        import glob
        import torch

        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from vggt_shapes import extract_shapes

        device = (
            torch.device("mps")  if torch.backends.mps.is_available()  else
            torch.device("cuda") if torch.cuda.is_available()           else
            torch.device("cpu")
        )
        print(f"[VGGTForwardPass] Loading model on {device} …")
        model = VGGT.from_pretrained("facebook/VGGT-1B")
        model = model.to(device).eval()

        img_dir = os.path.join(VGGT_PATH, "examples", "kitchen", "images")
        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))[:3]
        images = load_and_preprocess_images(img_paths, mode="crop")
        images_dev = images.to(device)

        shapes = extract_shapes(model, images_dev)
        print(f"[VGGTForwardPass] Captured {len(shapes)} shapes.")
        return shapes
    except Exception as exc:
        print(f"[VGGTForwardPass] Could not load model: {exc}")
        print("[VGGTForwardPass] Using placeholder shapes for animation.")
        return {}


def _default_shapes() -> dict:
    """Fallback shapes matching pre-agreed schema (kitchen 3-frame, 350×518)."""
    return {
        "input":              (1, 3, 3, 350, 518),
        "patch_embed":        (3, 925, 1024),
        "tokens_concat":      (3, 930, 1024),
        "frame_block_first":  (3, 930, 1024),
        "frame_block_last":   (3, 930, 1024),
        "global_block_first": (1, 2790, 1024),
        "global_block_last":  (1, 2790, 1024),
        "aggregated":         (1, 3, 930, 2048),
        "patch_start_idx":    5,
        "dpt_project_0":      (3,  256, 25, 37),
        "dpt_project_1":      (3,  512, 25, 37),
        "dpt_project_2":      (3, 1024, 25, 37),
        "dpt_project_3":      (3, 1024, 25, 37),
        "dpt_resize_0":       (3,  256, 100, 148),
        "dpt_resize_1":       (3,  512,  50,  74),
        "dpt_resize_2":       (3, 1024,  25,  37),
        "dpt_resize_3":       (3, 1024,  13,  19),
        "dpt_refinenet4":     (3,  256,  25,  37),
        "dpt_refinenet1":     (3,  256, 200, 296),
        "dpt_output":         (3,    2, 350, 518),
        "camera_token_norm":  (1, 3, 2048),
        "camera_trunk_last":  (1, 3, 2048),
        "camera_pose_branch": (1, 3, 9),
        "track_features":     (1, 3, 128, 175, 259),
    }


# ── Scene ─────────────────────────────────────────────────────────────────────

class VGGTForwardPass(Scene):
    """10-beat Manim animation of the VGGT-1B forward pass.

    Every Text label is populated from live hook-captured shapes.
    """

    def construct(self):
        # ── Beat 0: Load shapes ──────────────────────────────────────────────
        self.shapes = _load_shapes()
        if not self.shapes:
            self.shapes = _default_shapes()

        s = self.shapes  # shorthand

        # ── Beat 1: Title ────────────────────────────────────────────────────
        title = Text("VGGT Forward Pass", font_size=44, weight=BOLD)
        subtitle = Text("Visual Geometry Grounded Transformer — 1B params",
                        font_size=24, color=GRAY_C)
        subtitle.next_to(title, DOWN, buff=0.3)
        grp = VGroup(title, subtitle).move_to(ORIGIN)

        self.play(Write(title), run_time=1.2)
        self.play(FadeIn(subtitle, shift=UP * 0.2), run_time=0.8)
        self.wait(0.8)
        self.play(FadeOut(grp), run_time=0.5)

        # ── Beat 2: Input frames ──────────────────────────────────────────────
        inp_shape = s.get("input", (1, 3, 3, 350, 518))
        B, S_frames = inp_shape[0], inp_shape[1]
        H, W = inp_shape[-2], inp_shape[-1]

        frame_w, frame_h = 1.2, 0.85
        frames = VGroup(*[
            box(frame_w, frame_h, BLUE_BOX, f"t={i}", font_size=18)
            for i in range(S_frames)
        ]).arrange(RIGHT, buff=0.15)
        frames.move_to(LEFT * 4.5)

        inp_lbl = Text(f"Input  {list(inp_shape)}", font_size=18, color=SHAPE_COLOR)
        inp_lbl.next_to(frames, UP, buff=0.2)

        section = Text("① Input Frames", font_size=28, weight=BOLD, color=BLUE_D)
        section.to_edge(UP)
        self.play(Write(section), run_time=0.6)
        self.play(LaggedStartMap(FadeIn, frames, lag_ratio=0.15), run_time=1.0)
        self.play(Write(inp_lbl), run_time=0.5)
        self.wait(0.5)

        # ── Beat 3: Patch grid ───────────────────────────────────────────────
        ph = H // 14;  pw = W // 14
        patch_shape = s.get("patch_embed", (S_frames, ph * pw, 1024))
        N_patches = patch_shape[1]

        # Patch grid overlay on first frame rectangle
        first_frame = frames[0]
        grid_lines = VGroup()
        for col in range(1, min(pw, 8)):
            x = first_frame.get_left()[0] + col * frame_w / min(pw, 8)
            grid_lines.add(Line(
                [x, first_frame.get_bottom()[1], 0],
                [x, first_frame.get_top()[1], 0],
                stroke_width=0.5, color=BLUE_B
            ))
        for row in range(1, min(ph, 6)):
            y = first_frame.get_bottom()[1] + row * frame_h / min(ph, 6)
            grid_lines.add(Line(
                [first_frame.get_left()[0], y, 0],
                [first_frame.get_right()[0], y, 0],
                stroke_width=0.5, color=BLUE_B
            ))

        tok_seq = box(2.2, 0.5, BLUE_BOX, f"{N_patches} patch tokens", font_size=16)
        tok_seq.next_to(frames, RIGHT, buff=0.8)
        tok_lbl = shape_label(patch_shape)
        tok_lbl.next_to(tok_seq, DOWN, buff=0.1)

        sec2 = Text("② Patch Embedding", font_size=28, weight=BOLD, color=BLUE_D)
        sec2.to_edge(UP)
        self.play(Transform(section, sec2), run_time=0.4)
        self.play(Create(grid_lines), run_time=0.8)
        self.play(GrowArrow(Arrow(first_frame.get_right(), tok_seq.get_left(),
                                  buff=0.1, stroke_width=2)))
        self.play(FadeIn(tok_seq), Write(tok_lbl), run_time=0.6)
        self.wait(0.4)

        # ── Beat 4: Special tokens ───────────────────────────────────────────
        psi = s.get("patch_start_idx", 5)
        cam_tok = box(0.6, 0.5, GOLD_BOX, "cam", font_size=16)
        reg_toks = VGroup(*[
            box(0.5, 0.5, GRAY_BOX, "r", font_size=14)
            for _ in range(psi - 1)
        ]).arrange(RIGHT, buff=0.05)

        special = VGroup(cam_tok, reg_toks).arrange(RIGHT, buff=0.08)
        special.next_to(tok_seq, UP, buff=0.3)

        cat_box = box(3.5, 0.55, BLUE_BOX,
                      f"cat → {list(s.get('tokens_concat', (S_frames, N_patches+psi, 1024)))}",
                      font_size=15)
        cat_box.next_to(tok_seq, RIGHT, buff=0.6)

        sec3 = Text("③ Special Tokens + Concat", font_size=28, weight=BOLD, color=GOLD_E)
        sec3.to_edge(UP)
        self.play(Transform(section, sec3), run_time=0.3)
        self.play(FadeIn(special, shift=DOWN * 0.2), run_time=0.6)
        psi_lbl = Text(f"patch_start_idx = {psi}", font_size=16, color=SHAPE_COLOR)
        psi_lbl.next_to(special, RIGHT, buff=0.15)
        self.play(Write(psi_lbl), run_time=0.4)
        self.play(FadeIn(cat_box), run_time=0.5)
        self.wait(0.4)

        # Clear intermediate elements
        self.play(FadeOut(VGroup(frames, grid_lines, tok_seq, tok_lbl,
                                  special, psi_lbl, cat_box, inp_lbl)), run_time=0.5)

        # ── Beat 5: Frame attention ──────────────────────────────────────────
        fb_shape = s.get("frame_block_first", (S_frames, N_patches + psi, 1024))
        frame_blocks_vis = VGroup(*[
            box(1.0, 2.0, BLUE_BOX, f"S{i}", font_size=16)
            for i in range(min(S_frames, 4))
        ]).arrange(RIGHT, buff=0.6)
        frame_blocks_vis.move_to(ORIGIN)

        # Self-attention arcs within each frame
        attn_arcs = VGroup()
        for blk in frame_blocks_vis:
            arc = CurvedArrow(
                blk.get_top() + LEFT * 0.2,
                blk.get_top() + RIGHT * 0.2,
                angle=-TAU / 4, color=BLUE_B, stroke_width=1.5
            )
            attn_arcs.add(arc)

        fb_lbl = shape_label(fb_shape, font_size=16)
        fb_lbl.next_to(frame_blocks_vis, DOWN, buff=0.3)

        sec4 = Text("④ Frame Attention ×24", font_size=28, weight=BOLD, color=BLUE_D)
        sec4.to_edge(UP)
        self.play(Transform(section, sec4), run_time=0.3)
        self.play(LaggedStartMap(FadeIn, frame_blocks_vis, lag_ratio=0.1), run_time=0.8)
        self.play(LaggedStartMap(Create, attn_arcs, lag_ratio=0.05), run_time=0.8)
        self.play(Write(fb_lbl), run_time=0.4)
        self.wait(0.5)

        # ── Beat 6: Global attention ─────────────────────────────────────────
        gb_shape = s.get("global_block_first", (B, S_frames * (N_patches + psi), 1024))
        cross_arrows = VGroup()
        blk_list = list(frame_blocks_vis)
        for i in range(len(blk_list) - 1):
            arr = Arrow(blk_list[i].get_right(), blk_list[i + 1].get_left(),
                        buff=0.05, stroke_width=1.5, color=GREEN_D)
            cross_arrows.add(arr)

        gb_lbl = shape_label(gb_shape, font_size=16)
        gb_lbl.next_to(frame_blocks_vis, UP, buff=0.3)

        sec5 = Text("⑤ Global Attention ×24", font_size=28, weight=BOLD, color=GREEN_D)
        sec5.to_edge(UP)
        self.play(Transform(section, sec5), run_time=0.3)
        self.play(LaggedStartMap(GrowArrow, cross_arrows, lag_ratio=0.1), run_time=0.8)
        self.play(Write(gb_lbl), run_time=0.4)
        self.wait(0.5)

        # ── Beat 7: Aggregated features ──────────────────────────────────────
        agg_shape = s.get("aggregated", (B, S_frames, N_patches + psi, 2048))
        frame_blk = box(1.0, 1.2, BLUE_BOX, "frame\n1024", font_size=15)
        glob_blk  = box(1.0, 1.2, GREEN_BOX, "global\n1024", font_size=15)
        concat_grp = VGroup(frame_blk, glob_blk).arrange(RIGHT, buff=0.1)
        concat_grp.move_to(ORIGIN)

        plus = Text("‖", font_size=30, color=WHITE).next_to(frame_blk, RIGHT, buff=0.12)

        agg_out = box(2.4, 0.7, ORANGE_BOX, "concat → 2048", font_size=18)
        agg_out.next_to(concat_grp, DOWN, buff=0.6)
        agg_lbl = shape_label(agg_shape, font_size=16)
        agg_lbl.next_to(agg_out, DOWN, buff=0.1)

        sec6 = Text("⑥ Aggregated Features", font_size=28, weight=BOLD, color=ORANGE)
        sec6.to_edge(UP)
        self.play(Transform(section, sec6), run_time=0.3)
        self.play(FadeOut(VGroup(frame_blocks_vis, attn_arcs, cross_arrows,
                                  fb_lbl, gb_lbl)), run_time=0.4)
        self.play(FadeIn(concat_grp), Write(plus), run_time=0.7)
        self.play(GrowArrow(Arrow(concat_grp.get_bottom(), agg_out.get_top(), buff=0.05,
                                   stroke_width=2)))
        self.play(FadeIn(agg_out), Write(agg_lbl), run_time=0.5)
        self.wait(0.5)

        # ── Beat 8: Head branching ───────────────────────────────────────────
        sec7 = Text("⑦ Prediction Heads", font_size=28, weight=BOLD, color=WHITE)
        sec7.to_edge(UP)
        self.play(Transform(section, sec7), run_time=0.3)
        self.play(FadeOut(VGroup(concat_grp, plus)), run_time=0.3)
        agg_out.generate_target()
        agg_out.target.move_to(LEFT * 4.5)
        agg_lbl.generate_target()
        agg_lbl.target.next_to(agg_out.target, DOWN, buff=0.1)
        self.play(MoveToTarget(agg_out), MoveToTarget(agg_lbl), run_time=0.5)

        # Head boxes
        cam_box  = box(2.2, 0.65, GOLD_BOX,   "CameraHead",    font_size=20)
        dptD_box = box(2.2, 0.65, BLUE_BOX,   "DPTHead(depth)",font_size=20)
        dptP_box = box(2.2, 0.65, GREEN_BOX,  "DPTHead(point)",font_size=20)
        trk_box  = box(2.2, 0.65, DASHED_BOX, "TrackHead*",    font_size=20)

        heads = VGroup(cam_box, dptD_box, dptP_box, trk_box)
        heads.arrange(DOWN, buff=0.25)
        heads.move_to(RIGHT * 2.0)

        # Arrows: solid for cam/depth/point, dashed for track
        arrows = VGroup()
        src = agg_out.get_right()
        for i, hbox in enumerate(heads):
            dst = hbox.get_left()
            if i < 3:
                arr = Arrow(src, dst, buff=0.08, stroke_width=2, color=WHITE)
            else:
                arr = DashedLine(src, dst, dash_length=0.12, stroke_width=1.5, color=GRAY_C)
                arr = VGroup(arr, Arrow(dst + LEFT * 0.01, dst, buff=0, stroke_width=1.5,
                                        color=GRAY_C, max_tip_length_to_length_ratio=0.3))
            arrows.add(arr)

        self.play(LaggedStartMap(FadeIn, heads, lag_ratio=0.15), run_time=0.8)
        self.play(LaggedStartMap(Create, arrows, lag_ratio=0.1), run_time=0.8)
        self.wait(0.4)

        # ── Beat 9: Head output labels ────────────────────────────────────────
        pose_shape = s.get("camera_pose_branch", (B, S_frames, 9))
        dout_shape = s.get("dpt_output",         (S_frames, 2, H, W))
        trkf_shape = s.get("track_features",     (B, S_frames, 128, H // 2, W // 2))

        out_labels = VGroup(
            shape_label(pose_shape, 16).next_to(cam_box,  RIGHT, buff=0.25),
            shape_label(dout_shape,  16).next_to(dptD_box, RIGHT, buff=0.25),
            shape_label(dout_shape,  16).next_to(dptP_box, RIGHT, buff=0.25),
            shape_label(trkf_shape,  16).next_to(trk_box,  RIGHT, buff=0.25),
        )
        trk_note = Text("(optional: needs query_points)", font_size=13, color=GRAY_C)
        trk_note.next_to(trk_box, DOWN, buff=0.1)

        sec8 = Text("⑧ Outputs", font_size=28, weight=BOLD, color=YELLOW)
        sec8.to_edge(UP)
        self.play(Transform(section, sec8), run_time=0.3)
        self.play(LaggedStartMap(Write, out_labels, lag_ratio=0.15), run_time=1.0)
        self.play(FadeIn(trk_note), run_time=0.4)
        self.wait(1.0)

        # ── Beat 10: Fade out to summary ─────────────────────────────────────
        everything = VGroup(agg_out, agg_lbl, heads, arrows, out_labels,
                             trk_note, section)
        self.play(FadeOut(everything), run_time=0.6)

        summary = VGroup(
            Text("VGGT-1B Architecture", font_size=36, weight=BOLD),
            Text("Aggregator  →  CameraHead | DPTHead(depth) | DPTHead(point) | TrackHead",
                 font_size=20, color=GRAY_C),
            Text(f"Input {list(s.get('input',(1,3,3,350,518)))}  →  agg {list(s.get('aggregated',(1,3,930,2048)))}",
                 font_size=18, color=SHAPE_COLOR),
        ).arrange(DOWN, buff=0.35).move_to(ORIGIN)

        self.play(LaggedStartMap(FadeIn, summary, lag_ratio=0.2), run_time=1.2)
        self.wait(1.5)
        self.play(FadeOut(summary), run_time=0.8)


# ── CLI hint ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Render with:")
    print("  uv run manim -qm visualizations/vggt_manim.py VGGTForwardPass")
