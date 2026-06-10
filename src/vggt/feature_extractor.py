"""VGGT feature extractor for streaming 3D point extraction during RL acting."""

import sys
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# Ensure InfiniteVGGT source is importable.
_IVGGT_SRC = str(Path(__file__).resolve().parent.parent.parent / "external" / "InfiniteVGGT" / "src")
if _IVGGT_SRC not in sys.path:
    sys.path.insert(0, _IVGGT_SRC)

from streamvggt.models.streamvggt import StreamVGGT  # noqa: E402


# HuggingFace repo for the pretrained StreamVGGT checkpoint.
_HF_REPO = "lch01/StreamVGGT"

# Patch grid size: 518 / 14 = 37 patches per side.
_PATCH_GRID = 37


class VGGTFeatureExtractor:
    """Wraps InfiniteVGGT for streaming feature extraction during RL acting.

    The model is frozen and runs entirely under ``torch.no_grad()``.
    Call :meth:`reset` at every episode boundary to clear the KV-cache.
    Then call :meth:`extract` once per step with the current 518x518 RGB frame.
    """

    def __init__(
        self,
        device: str = "cuda",
        compile: bool = False,
        compile_mode: str | None = None,
        total_budget: int | None = None,
        max_camera_frames: int | None = None,
        camera_iterations: int = 4,
        prefer_flash_sdp: bool = True,
    ):
        """Load frozen InfiniteVGGT model.

        Args:
            device: Torch device string (default ``"cuda"``).
            compile: If True, wrap aggregator / camera_head / point_head with
                ``torch.compile``. Aggregator and camera_head use ``dynamic=True``
                (their KV-cache grows each frame). point_head is static.
                First few calls are slow (tracing); steady-state may be faster.
            compile_mode: Optional ``mode=`` argument forwarded to
                ``torch.compile`` when ``compile=True``. ``None`` (default)
                preserves the original behaviour (mode unset → torch's
                ``"default"``). Accepted values are ``None``, ``"default"``,
                ``"reduce-overhead"``, and ``"max-autotune"``.
                ``"max-autotune"`` can take minutes on the first call while
                kernels are autotuned.
            total_budget: Optional override for aggregator cache budget.
            max_camera_frames: Optional cap on camera-head KV cache length
                (in frames). The cache keeps the most recent frames.
            camera_iterations: Number of pose refinement iterations per frame
                (default 4) used to estimate camera-head cache growth.
            prefer_flash_sdp: If True, prefer Flash SDP kernels when available.
        """
        self.device = torch.device(device)

        # Load model via HuggingFace Hub (PyTorchModelHubMixin).
        self.model: StreamVGGT = StreamVGGT.from_pretrained(_HF_REPO)
        self.model = self.model.to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False

        if total_budget is not None:
            self.model.total_budget = int(total_budget)

        # Prefer Flash SDP kernels when available (PyTorch 2.0+).
        if prefer_flash_sdp and self.device.type == "cuda":
            self._flash_sdp_ctx_factory = (
                lambda: sdpa_kernel(
                    [
                        SDPBackend.FLASH_ATTENTION,
                        SDPBackend.EFFICIENT_ATTENTION,
                        SDPBackend.MATH,
                    ]
                )
            )
        else:
            self._flash_sdp_ctx_factory = lambda: nullcontext()
        self._prefer_flash_sdp = prefer_flash_sdp

        # Camera cache management (keep most recent frames).
        self._camera_iterations = int(camera_iterations)
        if max_camera_frames is not None:
            self._max_camera_tokens = int(max_camera_frames) * self._camera_iterations
        else:
            self._max_camera_tokens = None

        if compile:
            _allowed_modes = {None, "default", "reduce-overhead", "max-autotune"}
            if compile_mode not in _allowed_modes:
                raise ValueError(
                    f"compile_mode={compile_mode!r} not in {_allowed_modes}"
                )
            mode_kwargs = {} if compile_mode is None else {"mode": compile_mode}
            print(
                f"Compiling VGGT sub-modules with torch.compile "
                f"(mode={compile_mode!r})..."
            )
            self.model.aggregator = torch.compile(
                self.model.aggregator, dynamic=True, **mode_kwargs
            )
            self.model.camera_head = torch.compile(
                self.model.camera_head, dynamic=True, **mode_kwargs
            )
            self.model.point_head = torch.compile(
                self.model.point_head, **mode_kwargs
            )

        # Aggregator depth drives the per-layer KV-cache list length.
        self._agg_depth: int = self.model.aggregator.depth
        # Camera head trunk depth drives camera KV-cache list length.
        self._cam_depth: int = self.model.camera_head.trunk_depth
        # Max camera tokens for cache (tokens, not frames).
        self._max_camera_tokens: int | None = self._max_camera_tokens

        # Determine mixed-precision dtype based on GPU capability.
        if self.device.type == "cuda":
            cap = torch.cuda.get_device_capability(self.device)
            self._amp_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        else:
            self._amp_dtype = torch.float32

        # Initialise streaming state.
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear KV-cache and frame counter.  Call at episode boundaries."""
        self._past_key_values: list = [None] * self._agg_depth
        self._past_key_values_camera: list = [None] * self._cam_depth
        self._frame_idx: int = 0
        torch.cuda.empty_cache()

    def extract(
        self,
        rgb: np.ndarray,
        phase_times: dict[str, list[float]] | None = None,
    ) -> dict[str, np.ndarray]:
        """Single-frame streaming inference.

        Args:
            rgb: ``(3, 518, 518)`` uint8 array in CHW format (e.g. from Habitat).
            phase_times: Optional profiling hook. When provided, records per-call
                CUDA-Event durations (ms) under keys ``"vggt_forward"`` (input prep
                + model forward) and ``"vggt_wrapper"`` (pool + permute + .cpu()
                transfer). When ``None`` (default, production path) no events are
                created and no extra work is done.

        Returns:
            ``{"world_points": (37, 37, 3) float32,
              "camera_pose": (9,) float32}``
        """
        profiling = phase_times is not None
        if profiling:
            fwd_start = torch.cuda.Event(enable_timing=True)
            fwd_end = torch.cuda.Event(enable_timing=True)
            wrap_start = torch.cuda.Event(enable_timing=True)
            wrap_end = torch.cuda.Event(enable_timing=True)
            fwd_start.record()

        # --- prepare input tensor -------------------------------------------
        # uint8 CHW → float32 [0, 1], add batch dim → (1, 3, 518, 518)
        img = torch.from_numpy(rgb).to(dtype=torch.float32, device=self.device) / 255.0
        if img.dim() == 3:
            img = img.unsqueeze(0)  # (1, C, H, W)

        # --- streaming inference --------------------------------------------
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=self._amp_dtype):
            # Aggregator forward (single frame, KV-cached).
            # Images must be (B, S, C, H, W) — S=1 for streaming.
            images = img.unsqueeze(1)  # (1, 1, C, H, W)

            with self._flash_sdp_ctx_factory():
                aggregator_out = self.model.aggregator(
                    images,
                    past_key_values=self._past_key_values,
                    use_cache=True,
                    past_frame_idx=self._frame_idx,
                    total_budget=self.model.total_budget,
                )
            aggregated_tokens, patch_start_idx, self._past_key_values = aggregator_out

            # Camera head (with its own KV-cache).
            with torch.amp.autocast("cuda", enabled=False):
                with self._flash_sdp_ctx_factory():
                    pose_enc, self._past_key_values_camera = self.model.camera_head(
                        aggregated_tokens,
                        past_key_values_camera=self._past_key_values_camera,
                        use_cache=True,
                    )
                pose_enc = pose_enc[-1]           # last iteration
                camera_pose = pose_enc[:, 0, :]   # (B, 9)

                # Prune camera-head KV cache to a fixed window if requested.
                if self._max_camera_tokens is not None:
                    for idx, kv in enumerate(self._past_key_values_camera):
                        if kv is None:
                            continue
                        k, v = kv
                        if k.shape[2] > self._max_camera_tokens:
                            k = k[:, :, -self._max_camera_tokens :, :].contiguous()
                            v = v[:, :, -self._max_camera_tokens :, :].contiguous()
                            self._past_key_values_camera[idx] = (k, v)

                # Point head (DPT upsampled).
                pts3d, _ = self.model.point_head(
                    aggregated_tokens, images=images, patch_start_idx=patch_start_idx,
                )
                pts3d = pts3d[:, 0]  # (B, H, W, 3)  — remove sequence dim

        if profiling:
            fwd_end.record()
            wrap_start.record()

        # --- downsample to patch grid ----------------------------------------
        # pts3d is (1, H, W, 3) with H=W=518.  Pool to 37×37.
        # adaptive_avg_pool2d works on (N, C, H, W) so permute channels.
        pts_chw = pts3d.permute(0, 3, 1, 2).float()  # (1, 3, H, W)
        pts_down = F.adaptive_avg_pool2d(pts_chw, (_PATCH_GRID, _PATCH_GRID))  # (1, 3, 37, 37)
        world_points = pts_down.permute(0, 2, 3, 1).squeeze(0)  # (37, 37, 3)

        # --- to numpy --------------------------------------------------------
        world_points_np = world_points.cpu().float().numpy()
        camera_pose_np = camera_pose.squeeze(0).cpu().float().numpy()

        if profiling:
            wrap_end.record()
            torch.cuda.synchronize()
            phase_times["vggt_forward"].append(fwd_start.elapsed_time(fwd_end))
            phase_times["vggt_wrapper"].append(wrap_start.elapsed_time(wrap_end))

        # --- bookkeeping -----------------------------------------------------
        self._frame_idx += 1

        # === BP #1 (debug walkthrough — uncomment to re-enable) ===
        # print(f"\n[BP#1] frame_idx (post-incr)={self._frame_idx}", flush=True)
        # print(f"[BP#1] world_points: shape={world_points_np.shape} dtype={world_points_np.dtype}", flush=True)
        # print(f"[BP#1] world_points stats: "
        #       f"min={world_points_np.min():.3f} max={world_points_np.max():.3f} "
        #       f"mean={world_points_np.mean():.3f} std={world_points_np.std():.3f}", flush=True)
        # print(f"[BP#1] per-axis ranges: "
        #       f"x[{world_points_np[..., 0].min():.2f},{world_points_np[..., 0].max():.2f}] "
        #       f"y[{world_points_np[..., 1].min():.2f},{world_points_np[..., 1].max():.2f}] "
        #       f"z[{world_points_np[..., 2].min():.2f},{world_points_np[..., 2].max():.2f}]", flush=True)
        # print(f"[BP#1] NaN={np.isnan(world_points_np).any()} Inf={np.isinf(world_points_np).any()}", flush=True)
        # print(f"[BP#1] camera_pose: shape={camera_pose_np.shape} values={camera_pose_np}", flush=True)
        return {
            "world_points": world_points_np,   # (37, 37, 3) float32
            "camera_pose": camera_pose_np,     # (9,)        float32
        }
