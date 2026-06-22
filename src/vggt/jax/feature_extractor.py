"""JAX StreamVGGT feature extractor for streaming 3D point extraction.

Mirrors the public API of ``src.vggt.reference.feature_extractor.VGGTFeatureExtractor``
so callers can swap backends by changing one import:

    extractor = JAXVGGTFeatureExtractor(device="cuda")
    extractor.reset()                       # at every episode boundary
    out = extractor.extract(rgb)            # rgb: (3, 518, 518) uint8

Weights are loaded from the same HuggingFace checkpoint as the PyTorch
extractor, transposed into a Flax PyTree. The aggregator + camera-head
caches are carried as instance state across ``extract`` calls and cleared
by ``reset``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

# JAX compilation cache: re-use compiled graphs across runs / processes.
# Must be set before JAX touches the GPU.
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/vggt_jax_cache")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from src.vggt.jax.aggregator import (  # noqa: E402
    Aggregator,
    _calculate_dynamic_budgets,
)
from src.vggt.jax.heads.camera_head import CameraHead  # noqa: E402
from src.vggt.jax.heads.dpt_head import DPTHead  # noqa: E402
from src.vggt.jax.weight_transfer import (  # noqa: E402
    load_checkpoint,
    load_pytorch_weights,
)


# Image and patch grid are fixed at 518 / 14 = 37 patches per side.
_IMG_SIZE = 518
_PATCH_GRID = 37
_PATCH_SIZE = 14
# Reference default (streamvggt.models.streamvggt.StreamVGGT.__init__).
_DEFAULT_TOTAL_BUDGET = 1_200_000

ParamTree = dict[str, Any]
CacheEntry = tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
CompactCacheEntry = tuple[jnp.ndarray, jnp.ndarray]
PhaseTimes = dict[str, list[float]]
ExtractOutput = dict[str, jnp.ndarray]


@dataclass(frozen=True)
class ExtractorParams:
    """Flax parameter groups consumed by the three JIT-wrapped modules."""

    aggregator: ParamTree
    camera_head: ParamTree
    point_head: ParamTree


@dataclass(frozen=True)
class HeadOutputs:
    """Post-processed camera/point-head outputs for one streamed frame."""

    world_points: jnp.ndarray
    camera_pose: jnp.ndarray
    dense_world_points: jnp.ndarray | None


def select_jax_device(device: str) -> Any:
    """Resolve the public device string to a concrete JAX device."""
    if device in ("cuda", "gpu"):
        return jax.devices("gpu")[0]
    if device == "cpu":
        return jax.devices("cpu")[0]
    raise ValueError(f"unknown device {device!r}")


def load_params_on_device(device: Any) -> ExtractorParams:
    """Load StreamVGGT weights, convert them to Flax layout, and place on device."""
    state_dict = load_checkpoint()
    tree, _ = load_pytorch_weights(state_dict, include_v1_only=True)
    tree = jax.tree.map(lambda x: jax.device_put(jnp.asarray(x), device), tree)
    return ExtractorParams(
        aggregator={"params": tree["aggregator"]},
        camera_head={"params": tree["camera_head"]},
        point_head={"params": tree["point_head"]},
    )


def compile_point_head_apply(point_head: Any) -> Any:
    """JIT the point-head apply with static ``patch_start_idx``."""

    def _point_head_fn(
        params: ParamTree,
        out_list: list[jnp.ndarray],
        images: jnp.ndarray,
        patch_start_idx: int,
    ) -> tuple[jnp.ndarray, Any]:
        return point_head.apply(params, out_list, images, patch_start_idx)

    return jax.jit(_point_head_fn, static_argnums=(3,))


def compile_aggregator_apply(aggregator: Any) -> Any:
    """JIT one streaming aggregator step with static cache-control arguments."""

    def _agg_fn(
        params: ParamTree,
        images: jnp.ndarray,
        past_kvs: list[CacheEntry],
        is_first_frame: bool,
        total_budget: int,
        last_scores: jnp.ndarray,
        use_cache: bool,
        current_budgets_static: tuple[int, ...],
    ) -> tuple[list[jnp.ndarray], jnp.ndarray, list[CacheEntry], jnp.ndarray]:
        return aggregator.apply(
            params,
            images,
            use_cache=use_cache,
            past_kvs=past_kvs,
            past_frame_idx=0 if is_first_frame else 1,
            total_budget=total_budget,
            last_scores=last_scores,
            current_budgets_static=current_budgets_static,
        )

    return jax.jit(_agg_fn, static_argnums=(3, 4, 6, 7))


def compile_camera_head_apply(camera_head: Any) -> Any:
    """JIT the camera head against padded cache entries with stable shapes."""

    def _cam_fn(
        params: ParamTree,
        out_list: list[jnp.ndarray],
        past_kvs_camera: list[CacheEntry],
    ) -> tuple[list[jnp.ndarray], list[CacheEntry]]:
        return camera_head.apply(
            params,
            out_list,
            use_cache=True,
            past_kvs_camera=past_kvs_camera,
        )

    return jax.jit(_cam_fn)


def _pool_dense_world_points(pts_nhwc: jnp.ndarray, out_size: int) -> jnp.ndarray:
    """Average-pool ``(N, 518, 518, C)`` to ``(N, out_size, out_size, C)``.

    Divisible sizes use exact block means, e.g. ``37`` maps to 14x14 cells.
    Non-divisible sizes use antialiased area-style resizing for configurable
    WP+CP readouts such as the 64x64 ablation.
    """
    n_batch, height, width, channels = pts_nhwc.shape
    if (height, width) != (_IMG_SIZE, _IMG_SIZE):
        raise ValueError(
            f"expected ({_IMG_SIZE}, {_IMG_SIZE}), got ({height}, {width})"
        )
    if out_size <= 0 or out_size > _IMG_SIZE:
        raise ValueError(f"out_size must be in [1, {_IMG_SIZE}], got {out_size}")
    if _IMG_SIZE % out_size == 0:
        factor = _IMG_SIZE // out_size
        return pts_nhwc.reshape(
            n_batch,
            out_size,
            factor,
            out_size,
            factor,
            channels,
        ).mean(axis=(2, 4))
    return jax.image.resize(
        pts_nhwc,
        (n_batch, out_size, out_size, channels),
        method="linear",
        antialias=True,
    )


def _adaptive_avg_pool_518_to_37(pts_nhwc: jnp.ndarray) -> jnp.ndarray:
    """Average-pool ``(N, 518, 518, C)`` to exact 37x37 patch-grid cells."""
    return _pool_dense_world_points(pts_nhwc, _PATCH_GRID)


class JAXVGGTFeatureExtractor:
    """Drop-in JAX backend for ``VGGTFeatureExtractor``.

    The KV-cache is carried as instance state; call :meth:`reset` at every
    episode boundary. Internally the aggregator cache is stored in padded
    form (3-tuples of ``(k, v, valid_len)``) for JIT stability; the
    ``_past_kvs`` property exposes a compact 2-tuple view for tests.
    """

    def __init__(
        self,
        device: str = "cuda",
        compile: bool = False,
        total_budget: int = _DEFAULT_TOTAL_BUDGET,
        dtype: Any = jnp.bfloat16,
        max_camera_frames: int = 1024,
        budgets_static: tuple[int, ...] | None = None,
        compute_heads: bool = True,
        wp_pool_size: int = _PATCH_GRID,
    ) -> None:
        """Initialize weights, cache dimensions, JIT callables, and warmup graphs."""
        self._configure_runtime_options(compute_heads, wp_pool_size, budgets_static)
        self._device = select_jax_device(device)
        self._compile = compile  # reserved
        self._total_budget = total_budget
        self._dtype = dtype

        params = load_params_on_device(self._device)
        self._agg_params = params.aggregator
        self._cam_params = params.camera_head
        self._pt_params = params.point_head

        self._init_modules()
        self._configure_aggregator_cache(total_budget)
        self._finalize_static_budget_override()
        self._configure_camera_cache(max_camera_frames)
        self._compile_apply_functions()

        self.reset()
        self._warmup()

    def _configure_runtime_options(
        self,
        compute_heads: bool,
        wp_pool_size: int,
        budgets_static: tuple[int, ...] | None,
    ) -> None:
        """Store public options that alter output wrapping or cache budgets."""
        self._compute_heads = compute_heads
        self._wp_pool_size = int(wp_pool_size)
        self._budgets_static_override = budgets_static

    def _init_modules(self) -> None:
        """Construct module instances and cache their static depths."""
        self._aggregator = Aggregator()
        self._camera_head = CameraHead()
        self._point_head = DPTHead()

        self._agg_depth = self._aggregator.depth
        self._cam_depth = self._camera_head.trunk_depth

    def _configure_aggregator_cache(self, total_budget: int) -> None:
        """Derive padded cache dimensions for the streaming aggregator."""
        self._P = 5 + (_IMG_SIZE // _PATCH_SIZE) ** 2
        uniform = max(total_budget // self._agg_depth, self._P)
        self._MAX = uniform + self._P
        self._MAX_BUDGET = self._MAX - self._P
        self._num_heads = self._aggregator.num_heads
        self._head_dim = self._aggregator.embed_dim // self._num_heads

    def _finalize_static_budget_override(self) -> None:
        """Default to fixed budgets and validate explicit static budgets."""
        if self._MAX_BUDGET <= self._P:
            raise ValueError(
                f"total_budget={self._total_budget} leaves no room beyond "
                f"{self._P} anchor tokens per block"
            )
        if self._budgets_static_override is None:
            self._budgets_static_override = (self._MAX_BUDGET,) * self._agg_depth
            return
        if len(self._budgets_static_override) != self._agg_depth:
            raise ValueError(
                f"budgets_static length {len(self._budgets_static_override)} "
                f"!= aggregator depth {self._agg_depth}"
            )
        invalid = [
            budget
            for budget in self._budgets_static_override
            if budget > self._MAX_BUDGET or budget <= self._P
        ]
        if invalid:
            raise ValueError(
                "budgets_static entries must be in "
                f"({self._P}, {self._MAX_BUDGET}], got {invalid}"
            )

    def _configure_camera_cache(self, max_camera_frames: int) -> None:
        """Derive padded cache dimensions for the fixed-window camera head."""
        self._cam_num_heads = self._camera_head.num_heads
        self._cam_head_dim = self._camera_head.dim_in // self._cam_num_heads
        self._cam_num_iters = self._camera_head.num_iterations
        self._CAM_MAX = max_camera_frames * self._cam_num_iters

    def _compile_apply_functions(self) -> None:
        """Create JIT wrappers around the stateful module apply calls."""
        self._point_head_apply = compile_point_head_apply(self._point_head)
        self._aggregator_apply = compile_aggregator_apply(self._aggregator)
        self._camera_head_apply = compile_camera_head_apply(self._camera_head)

    @property
    def patch_grid(self) -> int:
        """Number of VGGT patch tokens per spatial side for the configured image size."""
        return _IMG_SIZE // _PATCH_SIZE

    @property
    def image_size(self) -> int:
        """Side length (pixels) of the square RGB input and the dense WP map."""
        return _IMG_SIZE

    @property
    def wp_pool_size(self) -> int:
        """Grid the dense point map is pooled to for the `world_points` output."""
        return self._wp_pool_size

    @property
    def aggregator_feature_shape(self) -> tuple[int, int]:
        """Shape of one frame's all-token pre-head global aggregator features."""
        return (
            1 + self._aggregator.num_register_tokens + self.patch_grid**2,
            self._aggregator.embed_dim,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _new_padded_cache_entry(self) -> CacheEntry:
        """Allocate a zero-padded (k, v, valid_len=0) entry."""
        B = 1
        k_pad = jnp.zeros(
            (B, self._num_heads, self._MAX, self._head_dim),
            dtype=self._dtype,
        )
        v_pad = jnp.zeros(
            (B, self._num_heads, self._MAX, self._head_dim),
            dtype=self._dtype,
        )
        valid_len = jnp.asarray(0, dtype=jnp.int32)
        return (k_pad, v_pad, valid_len)

    def _new_padded_camera_entry(self) -> CacheEntry:
        """Allocate a zero-padded camera-head (k, v, valid_len=0) entry."""
        B = 1
        k_pad = jnp.zeros(
            (B, self._cam_num_heads, self._CAM_MAX, self._cam_head_dim),
            dtype=self._dtype,
        )
        v_pad = jnp.zeros(
            (B, self._cam_num_heads, self._CAM_MAX, self._cam_head_dim),
            dtype=self._dtype,
        )
        valid_len = jnp.asarray(0, dtype=jnp.int32)
        return (k_pad, v_pad, valid_len)

    def _compute_static_budgets(self, last_scores_np: np.ndarray) -> tuple[int, ...]:
        """Compute dynamic per-block budgets as a tuple of Python ints."""
        bud = np.asarray(
            _calculate_dynamic_budgets(
                jnp.asarray(last_scores_np, dtype=jnp.float32),
                self._total_budget,
            )
        )
        # Leave room for the next frame append before eviction runs.
        bud = np.clip(bud, self._P + 1, self._MAX_BUDGET)
        return tuple(int(x) for x in bud.tolist())

    def _warmup(self) -> None:
        """Pre-compile both cache states so the first real call is fast."""
        dummy = jnp.zeros((1, 1, 3, _IMG_SIZE, _IMG_SIZE), dtype=self._dtype)

        # Frame 0: padded cache with valid_len=0 on every block.
        past0 = [self._new_padded_cache_entry() for _ in range(self._agg_depth)]
        last0 = jnp.zeros((self._agg_depth,), dtype=jnp.float32)
        bud0 = self._compute_static_budgets(np.zeros(self._agg_depth, dtype=np.float32))
        out0 = self._aggregator_apply(
            self._agg_params,
            dummy,
            past0,
            True,
            self._total_budget,
            last0,
            True,
            bud0,
        )
        out0[0][-1].block_until_ready()

        # Frame 1: reuse returned cache to compile the "is_first_frame=False" graph.
        _, _, past1, last1 = out0
        # Keep same budget (same last_scores pre-eviction) so shapes match.
        out1 = self._aggregator_apply(
            self._agg_params,
            dummy,
            past1,
            False,
            self._total_budget,
            last1,
            True,
            bud0,
        )
        out1[0][-1].block_until_ready()

        # Camera-head warmup: single graph covers all frames since the padded
        # cache keeps shapes stable regardless of valid_len.
        past_cam0 = [self._new_padded_camera_entry() for _ in range(self._cam_depth)]
        pose_list_w, _ = self._camera_head_apply(self._cam_params, out0[0], past_cam0)
        pose_list_w[-1].block_until_ready()

    # ------------------------------------------------------------------
    # Public cache view (_past_kvs property).
    # ------------------------------------------------------------------

    @property
    def _past_kvs(self) -> list[CompactCacheEntry] | None:
        """Compact 2-tuple view of the padded cache, for test observation.

        Returns None if no frames have been processed. Otherwise a list of
        ``(k, v)`` tuples, each shape ``(B, H, valid_len, Dh)``.
        """
        if self._past_kvs_padded is None:
            return None
        out: list[CompactCacheEntry] = []
        for entry in self._past_kvs_padded:
            k_pad, v_pad, valid_len = entry
            vl = int(np.asarray(valid_len))
            out.append((k_pad[:, :, :vl], v_pad[:, :, :vl]))
        return out

    @_past_kvs.setter
    def _past_kvs(
        self,
        value: list[CacheEntry | CompactCacheEntry | None] | None,
    ) -> None:
        """Setter supporting the ``self._past_kvs = None`` reset pattern.

        Writing anything non-None falls through to storing as padded state.
        In practice this is only used by ``reset()``.
        """
        if value is None:
            self._past_kvs_padded = None
        else:
            # Caller providing either 2-tuples or 3-tuples; re-pad as needed.
            self._past_kvs_padded = [self._to_padded(e) for e in value]

    def _to_padded(self, entry: CacheEntry | CompactCacheEntry | None) -> CacheEntry:
        """Convert a compact 2-tuple (or None) into padded 3-tuple form."""
        if entry is None:
            return self._new_padded_cache_entry()
        if isinstance(entry, tuple) and len(entry) == 3:
            return entry
        k, v = entry
        B, H, L, Dh = k.shape
        k_pad = jnp.zeros((B, H, self._MAX, Dh), dtype=k.dtype)
        v_pad = jnp.zeros((B, H, self._MAX, Dh), dtype=v.dtype)
        k_pad = jax.lax.dynamic_update_slice_in_dim(k_pad, k, 0, axis=2)
        v_pad = jax.lax.dynamic_update_slice_in_dim(v_pad, v, 0, axis=2)
        valid_len = jnp.asarray(L, dtype=jnp.int32)
        return (k_pad, v_pad, valid_len)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear KV-cache and frame counter. Call at episode boundaries."""
        self._past_kvs_padded: list[CacheEntry] | None = None
        self._last_scores: jnp.ndarray | None = None
        self._past_kvs_camera: list[CacheEntry] | None = None
        self._frame_idx: int = 0

    def _prepare_input_image(self, rgb: np.ndarray) -> jnp.ndarray:
        """Normalize a CHW uint8 frame and add batch/sequence dimensions."""
        if not isinstance(rgb, np.ndarray):
            raise TypeError(f"rgb must be a numpy array, got {type(rgb).__name__}")
        if rgb.shape != (3, _IMG_SIZE, _IMG_SIZE):
            raise ValueError(
                f"VGGT extractor expects CHW image shape "
                f"(3, {_IMG_SIZE}, {_IMG_SIZE}), got {rgb.shape}"
            )
        if rgb.dtype != np.uint8:
            raise ValueError(f"VGGT extractor expects uint8 image, got {rgb.dtype}")
        img = (jnp.asarray(rgb, dtype=jnp.float32) / 255.0).astype(self._dtype)
        return jax.device_put(img[None, None], self._device)

    def _ensure_aggregator_cache(self) -> None:
        """Allocate aggregator cache state lazily on the first frame after reset."""
        if self._past_kvs_padded is None:
            self._past_kvs_padded = [
                self._new_padded_cache_entry() for _ in range(self._agg_depth)
            ]
        if self._last_scores is None:
            self._last_scores = jnp.zeros((self._agg_depth,), dtype=jnp.float32)

    def _resolve_static_budgets(self) -> tuple[int, ...]:
        """Return per-block budgets as Python ints for JIT static args."""
        if self._budgets_static_override is not None:
            return self._budgets_static_override
        return self._compute_static_budgets(np.asarray(self._last_scores))

    def _run_aggregator(
        self,
        images: jnp.ndarray,
        budgets_static: tuple[int, ...],
    ) -> tuple[list[jnp.ndarray], jnp.ndarray]:
        """Run one streaming aggregator step and update aggregator cache state."""
        out_list, patch_start_idx, self._past_kvs_padded, self._last_scores = (
            self._aggregator_apply(
                self._agg_params,
                images,
                self._past_kvs_padded,
                self._frame_idx == 0,  # is_first_frame — bool, only 2 compiles total
                self._total_budget,
                self._last_scores,
                True,  # use_cache
                budgets_static,
            )
        )
        return out_list, patch_start_idx

    def _ensure_camera_cache(self) -> None:
        """Allocate camera-head cache state lazily when heads are enabled."""
        if self._past_kvs_camera is None:
            self._past_kvs_camera = [
                self._new_padded_camera_entry() for _ in range(self._cam_depth)
            ]

    def _check_camera_cache_capacity(self) -> None:
        """Fail before the next camera-head write would exceed the padded cache."""
        # Guard against silent cache overflow: dynamic_update_slice_in_dim
        # clamps out-of-range writes, and key_value_seq_lengths > _CAM_MAX
        # produces undefined cuDNN behavior.  Fail loud here instead.
        max_frames = self._CAM_MAX // self._cam_num_iters
        if self._frame_idx >= max_frames:
            raise RuntimeError(
                f"Camera-head padded cache overflow: cannot extract frame "
                f"{self._frame_idx + 1}, max_camera_frames={max_frames}. "
                f"Raise max_camera_frames at construction or call reset()."
            )

    def _run_heads(
        self,
        out_list: list[jnp.ndarray],
        images: jnp.ndarray,
        patch_start_idx: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        self._ensure_camera_cache()
        self._check_camera_cache_capacity()
        pose_list, self._past_kvs_camera = self._camera_head_apply(
            self._cam_params,
            out_list,
            self._past_kvs_camera,
        )
        camera_pose = pose_list[-1][:, 0, :]

        # patch_start_idx is always 1 + num_register_tokens = 5; cast to Python
        # int so it can be used as a static JIT arg (JIT returns JAX scalars).
        pts3d, _ = self._point_head_apply(
            self._pt_params,
            out_list,
            images,
            int(np.asarray(patch_start_idx)),
        )
        return pts3d[:, 0], camera_pose

    def _pool_head_outputs(
        self,
        pts3d: jnp.ndarray,
        camera_pose: jnp.ndarray,
        return_dense: bool,
    ) -> HeadOutputs:
        """Pool dense point maps and unwrap the single-frame camera pose."""
        world_points = _pool_dense_world_points(pts3d, self._wp_pool_size)
        world_points_out = world_points[0].astype(jnp.float32)
        camera_pose_out = camera_pose[0].astype(jnp.float32)
        # Pre-pool dense map (N, 518, 518, 3) -> (518, 518, 3); 3D-48.
        dense_world_points_out = pts3d[0].astype(jnp.float32) if return_dense else None
        return HeadOutputs(
            world_points=world_points_out,
            camera_pose=camera_pose_out,
            dense_world_points=dense_world_points_out,
        )

    def _aggregator_full_tokens(self, out_list: list[jnp.ndarray]) -> jnp.ndarray:
        """Expose final full-width frame+global aggregator tokens."""
        final_tokens = out_list[-1]
        return final_tokens[0, 0].astype(jnp.float32)

    def _aggregator_features(self, out_list: list[jnp.ndarray]) -> jnp.ndarray:
        """Expose the final global-stream tokens used by VGGT encoder variants."""
        # Final pre-head aggregator tokens for encoder ablations. The JAX port
        # stores frame/local and global/contextual streams concatenated as
        # 2048-d tokens for DPT heads; expose the 1024-d global stream requested
        # by Variant 1 before camera/point heads transform it into WP+CP. Keep
        # all VGGT-DP / VGGT-World tokens: camera + register + spatial patches.
        final_tokens = self._aggregator_full_tokens(out_list)
        return final_tokens[..., final_tokens.shape[-1] // 2 :]

    def _run_optional_heads(
        self,
        out_list: list[jnp.ndarray],
        images: jnp.ndarray,
        patch_start_idx: jnp.ndarray,
        *,
        return_dense: bool,
        phase_times: PhaseTimes | None,
        forward_start: float,
    ) -> HeadOutputs | None:
        """Run camera/point heads when enabled and record optional phase timings."""
        if not self._compute_heads:
            self._record_aggregator_only_profile(phase_times, forward_start)
            return None

        pts3d, camera_pose = self._run_heads(out_list, images, patch_start_idx)
        wrapper_start = self._synchronize_heads_for_profile(
            pts3d,
            camera_pose,
            phase_times,
        )
        head_outputs = self._pool_head_outputs(pts3d, camera_pose, return_dense)
        self._record_head_profile(phase_times, forward_start, wrapper_start)
        return head_outputs

    def _synchronize_heads_for_profile(
        self,
        pts3d: jnp.ndarray,
        camera_pose: jnp.ndarray,
        phase_times: PhaseTimes | None,
    ) -> float:
        """Synchronize head tensors only when wall-clock profiling is active."""
        if phase_times is None:
            return 0.0
        pts3d.block_until_ready()
        camera_pose.block_until_ready()
        return time.perf_counter()

    def _record_head_profile(
        self,
        phase_times: PhaseTimes | None,
        forward_start: float,
        wrapper_start: float,
    ) -> None:
        """Record forward and wrapper timings for the full extractor path."""
        if phase_times is None:
            return
        wrapper_end = time.perf_counter()
        phase_times["vggt_forward"].append((wrapper_start - forward_start) * 1000.0)
        phase_times["vggt_wrapper"].append((wrapper_end - wrapper_start) * 1000.0)

    def _record_aggregator_only_profile(
        self,
        phase_times: PhaseTimes | None,
        forward_start: float,
    ) -> None:
        """Record timings for compute_heads=False, where wrapper work is zero."""
        if phase_times is None:
            return
        forward_end = time.perf_counter()
        phase_times["vggt_forward"].append((forward_end - forward_start) * 1000.0)
        phase_times["vggt_wrapper"].append(0.0)

    def _build_extract_output(
        self,
        *,
        aggregator_full_tokens: jnp.ndarray,
        aggregator_features: jnp.ndarray,
        head_outputs: HeadOutputs | None,
        return_dense: bool,
    ) -> ExtractOutput:
        """Build the public output dict without changing legacy key names."""
        if self._compute_heads:
            if head_outputs is None:
                raise RuntimeError("head outputs missing while compute_heads=True")
            out = {
                "world_points": head_outputs.world_points,
                "camera_pose": head_outputs.camera_pose,
                "aggregator_features": aggregator_features,
                "aggregator_full_tokens": aggregator_full_tokens,
            }
            if return_dense:
                if head_outputs.dense_world_points is None:
                    raise RuntimeError("dense world points missing")
                out["dense_world_points"] = head_outputs.dense_world_points
            return out
        return {
            "aggregator_features": aggregator_features,
            "aggregator_full_tokens": aggregator_full_tokens,
        }

    def extract(
        self,
        rgb: np.ndarray,
        phase_times: PhaseTimes | None = None,
        return_dense: bool = False,
    ) -> ExtractOutput:
        """Single-frame streaming inference.

        When ``return_dense`` is True the result dict additionally carries
        ``dense_world_points`` — the pre-pool DPT point map at full
        518x518x3 resolution (one 3D point per pixel, the paper's
        "pixel-as-point" map). Diagnostic only (see issue 3D-48); the
        default path is unaffected and does not materialize it.
        """
        forward_start = time.perf_counter() if phase_times is not None else 0.0

        images = self._prepare_input_image(rgb)
        self._ensure_aggregator_cache()
        budgets_static = self._resolve_static_budgets()
        out_list, patch_start_idx = self._run_aggregator(images, budgets_static)

        head_outputs = self._run_optional_heads(
            out_list,
            images,
            patch_start_idx,
            return_dense=return_dense,
            phase_times=phase_times,
            forward_start=forward_start,
        )
        aggregator_full_tokens = self._aggregator_full_tokens(out_list)
        aggregator_features = aggregator_full_tokens[
            ..., aggregator_full_tokens.shape[-1] // 2 :
        ]
        self._frame_idx += 1
        return self._build_extract_output(
            aggregator_full_tokens=aggregator_full_tokens,
            aggregator_features=aggregator_features,
            head_outputs=head_outputs,
            return_dense=return_dense,
        )
