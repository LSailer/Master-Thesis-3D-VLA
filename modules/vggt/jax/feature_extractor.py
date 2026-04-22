"""JAX StreamVGGT feature extractor for streaming 3D point extraction.

Mirrors the public API of ``modules.vggt.feature_extractor.VGGTFeatureExtractor``
so callers can swap backends by changing one import:

    extractor = JAXVGGTFeatureExtractor(device="cuda")
    extractor.reset()                       # at every episode boundary
    out = extractor.extract(rgb)            # rgb: (3, 518, 518) uint8
    # out = {"world_points": (37, 37, 3) float32,
    #        "camera_pose":  (9,)        float32}

Weights are loaded from the same HuggingFace checkpoint as the PyTorch
extractor, transposed into a Flax PyTree. The aggregator + camera-head
caches are carried as instance state across ``extract`` calls and cleared
by ``reset``.

v1 runs fp32 end-to-end; bf16 autocast and ``jax.jit`` come in Step 8 with
the benchmark harness.
"""

from __future__ import annotations

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from modules.vggt.jax.aggregator import Aggregator
from modules.vggt.jax.heads.camera_head import CameraHead
from modules.vggt.jax.heads.dpt_head import DPTHead
from modules.vggt.jax.weight_transfer import load_checkpoint, load_pytorch_weights

# Image and patch grid are fixed at 518 / 14 = 37 patches per side.
_IMG_SIZE = 518
_PATCH_GRID = 37
_PATCH_SIZE = 14
# Reference default (streamvggt.models.streamvggt.StreamVGGT.__init__).
_DEFAULT_TOTAL_BUDGET = 1_200_000


def _adaptive_avg_pool_518_to_37(pts_nhwc: jnp.ndarray) -> jnp.ndarray:
    """Average-pool (N, 518, 518, C) to (N, 37, 37, C).

    ``F.adaptive_avg_pool2d(518→37)`` with integer ratio 14 is exactly
    non-overlapping 14x14 box pooling. Implemented as reshape + mean.
    """
    N, H, W, C = pts_nhwc.shape
    if (H, W) != (_IMG_SIZE, _IMG_SIZE):
        raise ValueError(f"expected (518, 518), got ({H}, {W})")
    out = pts_nhwc.reshape(
        N, _PATCH_GRID, _PATCH_SIZE, _PATCH_GRID, _PATCH_SIZE, C
    ).mean(axis=(2, 4))
    return out  # (N, 37, 37, C)


class JAXVGGTFeatureExtractor:
    """Drop-in JAX backend for ``VGGTFeatureExtractor``.

    Loads the InfiniteVGGT checkpoint once and runs streaming inference
    under eager JAX. The KV-cache (aggregator + camera-head) is carried as
    instance state; call :meth:`reset` at every episode boundary.
    """

    def __init__(
        self,
        device: str = "cuda",
        compile: bool = False,
        total_budget: int = _DEFAULT_TOTAL_BUDGET,
    ):
        """Load frozen StreamVGGT weights and prepare Flax modules.

        Args:
            device: Either ``"cuda"``/``"gpu"`` (JAX picks the first CUDA
                device) or ``"cpu"``. Present for PyTorch-API compatibility.
            compile: Accepted for API symmetry with the PyTorch extractor
                but ignored in v1. ``jax.jit`` is enabled by Step 8.
            total_budget: Global KV-cache budget fed to the aggregator's
                dynamic allocator. Matches the 1.2M default used by
                ``streamvggt.models.streamvggt.StreamVGGT``.
        """
        if device in ("cuda", "gpu"):
            self._device = jax.devices("gpu")[0]
        elif device == "cpu":
            self._device = jax.devices("cpu")[0]
        else:
            raise ValueError(f"unknown device {device!r}")
        self._compile = compile  # reserved
        self._total_budget = total_budget

        # Load weights: HF checkpoint -> numpy state_dict -> Flax PyTree.
        sd = load_checkpoint()
        tree, _ = load_pytorch_weights(sd, include_v1_only=True)
        tree = jax.tree.map(lambda x: jax.device_put(jnp.asarray(x), self._device), tree)
        self._agg_params = {"params": tree["aggregator"]}
        self._cam_params = {"params": tree["camera_head"]}
        self._pt_params = {"params": tree["point_head"]}

        # Bound module instances; Flax modules are cheap to construct.
        self._aggregator = Aggregator()
        self._camera_head = CameraHead()
        self._point_head = DPTHead()

        self._agg_depth = self._aggregator.depth
        self._cam_depth = self._camera_head.trunk_depth

        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear KV-cache and frame counter. Call at episode boundaries."""
        self._past_kvs: list[Any] | None = None
        self._last_scores: jnp.ndarray | None = None
        self._past_kvs_camera: list[Any] | None = None
        self._frame_idx: int = 0

    def extract(
        self,
        rgb: np.ndarray,
        phase_times: dict[str, list[float]] | None = None,
    ) -> dict[str, np.ndarray]:
        """Single-frame streaming inference.

        Args:
            rgb: ``(3, 518, 518)`` uint8 CHW (e.g. from Habitat).
            phase_times: Optional profiling hook. Records wall-clock ms under
                ``"vggt_forward"`` (input prep + model forward) and
                ``"vggt_wrapper"`` (pool + device-transfer).

        Returns:
            ``{"world_points": (37, 37, 3) float32,
               "camera_pose": (9,)        float32}``
        """
        profiling = phase_times is not None
        fwd_t0 = time.perf_counter() if profiling else 0.0

        # Input prep: uint8 CHW -> float32 [0, 1] -> (1, 1, 3, 518, 518)
        img = jnp.asarray(rgb, dtype=jnp.float32) / 255.0
        images = img[None, None]  # add batch + sequence dims
        images = jax.device_put(images, self._device)

        # Aggregator (streaming cache + dynamic budget eviction).
        out_list, patch_start_idx, self._past_kvs, self._last_scores = (
            self._aggregator.apply(
                self._agg_params,
                images,
                use_cache=True,
                past_kvs=self._past_kvs,
                past_frame_idx=self._frame_idx,
                total_budget=self._total_budget,
                last_scores=self._last_scores,
            )
        )

        # Camera head (iterative refiner with its own cache).
        pose_list, self._past_kvs_camera = self._camera_head.apply(
            self._cam_params,
            out_list,
            use_cache=True,
            past_kvs_camera=self._past_kvs_camera,
        )
        pose_enc = pose_list[-1]  # last refinement iteration
        camera_pose = pose_enc[:, 0, :]  # (B, 9)

        # Point head (DPT, no cache).
        pts3d, _ = self._point_head.apply(
            self._pt_params,
            out_list,
            images,
            patch_start_idx,
        )
        # pts3d is (B, S=1, H, W, 3); drop sequence dim.
        pts3d = pts3d[:, 0]

        if profiling:
            pts3d.block_until_ready()
            camera_pose.block_until_ready()
            wrap_t0 = time.perf_counter()

        # Pool 518x518 -> 37x37 via non-overlapping 14x14 mean.
        world_points = _adaptive_avg_pool_518_to_37(pts3d)  # (1, 37, 37, 3)

        world_points_np = np.asarray(world_points[0], dtype=np.float32)
        camera_pose_np = np.asarray(camera_pose[0], dtype=np.float32)

        if profiling:
            wrap_t1 = time.perf_counter()
            # Approximate the PyTorch hook's CUDA-event split; both phases
            # include the blocking wait via block_until_ready/asarray.
            phase_times["vggt_forward"].append((wrap_t0 - fwd_t0) * 1000.0)
            phase_times["vggt_wrapper"].append((wrap_t1 - wrap_t0) * 1000.0)

        self._frame_idx += 1

        return {
            "world_points": world_points_np,
            "camera_pose": camera_pose_np,
        }
