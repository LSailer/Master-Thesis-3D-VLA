"""vggt_shapes.py — PyTorch hook-based shape extractor for VGGT-1B.

Function: extract_shapes(model, images_dev) -> dict[str, tuple | int]

All shapes are captured from a single live forward pass via register_forward_hook.
No constants are hardcoded — every dimension comes directly from the model at runtime.
"""

import sys
import os

import torch


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _ensure_vggt_path():
    vggt_path = os.path.join(_repo_root(), "external", "VGGT")
    if vggt_path not in sys.path:
        sys.path.insert(0, vggt_path)


def extract_shapes(model, images_dev: torch.Tensor) -> dict:
    """Run one forward pass and collect tensor shapes at key VGGT hook points.

    Args:
        model: VGGT instance (eval mode, already on device).
        images_dev: [S,3,H,W] or [B,S,3,H,W] float tensor on same device.

    Returns:
        dict mapping string keys -> shape tuples (or int for patch_start_idx).
        Keys follow the pre-agreed schema used by vggt_arch.py / vggt_manim.py.
    """
    shapes: dict = {}
    hooks: list = []

    # ── Helper ───────────────────────────────────────────────────────────────

    def reg(module, key, selector=None):
        """Register a forward hook that stores output shape under `key`."""
        def _hook(m, inp, out):
            target = selector(out) if selector else out
            if isinstance(target, torch.Tensor):
                shapes[key] = tuple(target.shape)
        hooks.append(module.register_forward_hook(_hook))

    # DINOv2 patch embed returns a dict; pull out patch tokens.
    def _pe(out):
        return out["x_norm_patchtokens"] if isinstance(out, dict) else out

    # ── Aggregator ───────────────────────────────────────────────────────────

    reg(model.aggregator.patch_embed, "patch_embed", _pe)
    reg(model.aggregator.frame_blocks[0],  "frame_block_first")
    reg(model.aggregator.frame_blocks[-1], "frame_block_last")
    reg(model.aggregator.global_blocks[0],  "global_block_first")
    reg(model.aggregator.global_blocks[-1], "global_block_last")

    # ── DPT depth head ───────────────────────────────────────────────────────

    if model.depth_head is not None:
        for i in range(4):
            reg(model.depth_head.projects[i],      f"dpt_project_{i}")
            reg(model.depth_head.resize_layers[i], f"dpt_resize_{i}")
        reg(model.depth_head.scratch.refinenet4,    "dpt_refinenet4")
        reg(model.depth_head.scratch.refinenet1,    "dpt_refinenet1")
        reg(model.depth_head.scratch.output_conv2,  "dpt_output")

    # ── Camera head ──────────────────────────────────────────────────────────

    if model.camera_head is not None:
        reg(model.camera_head.token_norm,   "camera_token_norm")
        reg(model.camera_head.trunk[-1],    "camera_trunk_last")
        reg(model.camera_head.pose_branch,  "camera_pose_branch")

    # ── Track head (optional — only runs when query_points provided) ─────────

    if model.track_head is not None:
        reg(model.track_head.feature_extractor, "track_features")

    # ── Monkey-patch aggregator.forward to capture its return value ──────────
    # (output_list[-1] is the last of 24 frame∥global concat tensors [B,S,P,2C])

    orig_agg_forward = model.aggregator.forward

    def _patched_agg(images):
        out_list, psi = orig_agg_forward(images)
        shapes["aggregated"] = tuple(out_list[-1].shape)
        shapes["patch_start_idx"] = psi
        return out_list, psi

    model.aggregator.forward = _patched_agg

    # ── Build inputs ─────────────────────────────────────────────────────────

    imgs = images_dev
    if imgs.dim() == 4:
        imgs = imgs.unsqueeze(0)          # add batch dim → [B,S,3,H,W]

    shapes["input"] = tuple(imgs.shape)

    # Dummy query_points for track head (5 points at image centre-ish)
    B_t, S_t = imgs.shape[:2]
    qp = None
    if model.track_head is not None:
        H_t, W_t = imgs.shape[-2:]
        pts = torch.tensor([[H_t // 2, W_t // 2]], dtype=torch.float32,
                           device=imgs.device)
        qp = pts.unsqueeze(0).expand(B_t, 5, 2).clone()

    # ── Single forward pass ──────────────────────────────────────────────────

    with torch.no_grad():
        try:
            model(imgs, query_points=qp)
        except Exception as exc:  # tracker may crash on dummy pts — fine
            print(f"[vggt_shapes] forward raised {type(exc).__name__}: {exc}")

    # ── Restore & clean up ───────────────────────────────────────────────────

    model.aggregator.forward = orig_agg_forward
    for h in hooks:
        h.remove()

    # ── Derived shapes ───────────────────────────────────────────────────────

    if "patch_embed" in shapes and "patch_start_idx" in shapes:
        B_S, N_patches, C = shapes["patch_embed"]
        psi = shapes["patch_start_idx"]
        shapes["tokens_concat"] = (B_S, N_patches + psi, C)

    return shapes


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import glob

    _ensure_vggt_path()

    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images

    device = (
        torch.device("mps")  if torch.backends.mps.is_available()  else
        torch.device("cuda") if torch.cuda.is_available()           else
        torch.device("cpu")
    )
    print(f"Device: {device}")

    print("Loading VGGT-1B …")
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model = model.to(device).eval()

    _vggt = os.path.join(_repo_root(), "external", "VGGT")
    img_dir = os.path.join(_vggt, "examples", "kitchen", "images")
    img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))[:3]
    print(f"Using {len(img_paths)} kitchen frames: {[os.path.basename(p) for p in img_paths]}")

    images = load_and_preprocess_images(img_paths, mode="crop")
    images_dev = images.to(device)

    print("Extracting shapes …")
    shapes = extract_shapes(model, images_dev)

    print(f"\n{'Key':<28} {'Shape / Value':>30}")
    print("-" * 60)
    for k, v in sorted(shapes.items()):
        print(f"{k:<28} {str(v):>30}")
