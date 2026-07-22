"""JAX StreamVGGT feature extractor for streaming 3D point extraction.

Mirrors the public API of ``src.vggt.reference.feature_extractor.VGGTFeatureExtractor``
so callers can swap backends by changing one import:

    extractor = JAXVGGTFeatureExtractor(device="cuda")
    extractor.reset()                       # at every episode boundary
    out = extractor.extract(rgb)            # rgb: (518, 518, 3) uint8 HWC

Weights are loaded from the same HuggingFace checkpoint as the PyTorch
extractor, transposed into a Flax PyTree. The aggregator + camera-head
caches are carried as instance state across ``extract`` calls and cleared
by ``reset``.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

# JAX compilation cache: re-use compiled graphs across runs / processes.
# Must be set before JAX touches the GPU.
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/vggt_jax_cache")

# pylint: disable=wrong-import-position
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from src.environments.observation import ObservationFrame  # noqa: E402
from src.shared.ply_io import write_world_points_ply  # noqa: E402
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
# pylint: enable=wrong-import-position

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


@dataclass(frozen=True)
class VGGTExtractOutput:
    """Structured VGGT output for one streamed observation frame.

    ``world_points`` is the full-resolution point map when heads are enabled.
    ``confidence`` is the point head's per-pixel confidence for that map (the
    ``expp1`` activation, so values are ``>= 1``); it shares ``world_points``'
    spatial layout so the two flatten in lockstep. ``camera_pose`` is the final
    camera-head pose encoding. ``frame_tokens`` and ``global_tokens`` are the two
    halves of the final full-width aggregator tokens (all 1374 tokens of the
    last layer, float32 — the heads-OFF RSSM embedding surface).

    ``consumed_layer_halves`` is the heads-ON token-surgery surface (openspec
    change ``global-token-reconstruction-ablation``): for each DPT-consumed
    layer (``DPTHead.intermediate_layer_idx``, i.e. 4/11/17/23) the per-patch
    frame-half ``[:1024]`` and global-half ``[1024:]`` channel slices, each of
    shape ``(patch_grid**2, 1024)`` = ``(1369, 1024)`` in the retained
    aggregator dtype (bfloat16 — kept uncast so reassembled tokens reproduce
    ``point_head_from_tokens`` bit-for-bit). Camera/register tokens are
    dropped; a pooled scene vector is never produced.
    """

    world_points: jnp.ndarray
    confidence: jnp.ndarray
    camera_pose: jnp.ndarray
    frame_tokens: jnp.ndarray
    global_tokens: jnp.ndarray

    def __post_init__(self) -> None:
        """Normalizes point-map fields and validates their alignment.

        ``world_points`` and ``confidence`` are coerced to float32 ``jnp``
        arrays (via ``object.__setattr__``, since the dataclass is frozen), so
        consumers can use them without re-converting.

        Raises:
          ValueError: If confidence does not share world_points' spatial layout.
        """
        if self.world_points is None or self.confidence is None:
            return
        object.__setattr__(
            self, "world_points", jnp.asarray(self.world_points, dtype=jnp.float32)
        )
        object.__setattr__(
            self, "confidence", jnp.asarray(self.confidence, dtype=jnp.float32)
        )
        if self.world_points.shape[:-1] != self.confidence.shape:
            raise ValueError(
                "world_points/confidence spatial layout mismatch: "
                f"{self.world_points.shape} vs {self.confidence.shape}"
            )


ExtractOutput = VGGTExtractOutput


@dataclass(frozen=True)
class ExtractorParams:
    """Flax parameter groups consumed by the three JIT-wrapped modules."""

    aggregator: ParamTree
    camera_head: ParamTree
    point_head: ParamTree


@dataclass(frozen=True)
class HeadOutputs:
    """Post-processed head outputs for one streamed frame."""

    world_points: jnp.ndarray
    confidence: jnp.ndarray
    camera_pose: jnp.ndarray


class ResetMode(Enum):
    """How the extractor handles cache state at an episode/scene boundary.

    ``FULL`` wipes all streaming state every episode (the existing behaviour;
    safe, re-anchors each episode). ``PERSIST_SCENE`` saves the current cache
    keyed by ``scene_id`` and restores it when the same scene is seen again, so
    the VGGT attention stream resumes across episodes of one house instead of
    re-deriving geometry from scratch. The aggregator cache self-bounds via
    eviction, but the camera-head cache does not — see ``HANDOFF.md`` §2/§4.2
    before enabling ``PERSIST_SCENE`` for long runs.
    """

    FULL = "full"
    PERSIST_SCENE = "scene"


@dataclass(frozen=True)
class AggregatorCacheSnapshot:
    """Immutable snapshot of the four streaming-state fields.

    JAX arrays are immutable, so holding references is a true snapshot — the
    extractor reassigns these fields each frame (functional updates), it never
    mutates the arrays in place. ``save_cache``/``load_cache`` move these
    references in and out of the per-scene store without copying device memory.
    """

    past_kvs_padded: tuple[CacheEntry, ...] | None
    last_scores: jnp.ndarray | None
    past_kvs_camera: tuple[CacheEntry, ...] | None
    frame_idx: int


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
        enable_compile: bool = False,
        total_budget: int = _DEFAULT_TOTAL_BUDGET,
        dtype: Any = jnp.bfloat16,
        max_camera_frames: int = 1024,
        budgets_static: tuple[int, ...] | None = None,
        compute_heads: bool = True,
        wp_pool_size: int = _PATCH_GRID,
        reset_mode: ResetMode = ResetMode.FULL,
    ) -> None:
        """Initialize weights, cache dimensions, JIT callables, and warmup graphs."""
        self._configure_runtime_options(compute_heads, wp_pool_size, budgets_static)
        self._device = select_jax_device(device)
        self._enable_compile = enable_compile  # reserved
        self._total_budget = total_budget
        self._dtype = dtype
        self._reset_mode = reset_mode
        # Per-scene save/restore store for ResetMode.PERSIST_SCENE. Maps
        # scene_id -> snapshot of the streaming state at the last frame of that
        # scene. Only populated under PERSIST_SCENE; empty under FULL.
        self._scene_cache_store: dict[str, AggregatorCacheSnapshot] = {}
        self._current_scene_id: str | None = None

        params = load_params_on_device(self._device)
        self._agg_params = params.aggregator
        self._cam_params = params.camera_head
        self._pt_params = params.point_head

        self._init_modules()
        self._configure_aggregator_cache(total_budget)
        self._finalize_static_budget_override()
        self._configure_camera_cache(max_camera_frames)
        self._compile_apply_functions()

        # Last-frame aggregator outputs + raw image, retained so an occasional
        # PLY snapshot (write_point_cloud_ply) can run the point head without
        # re-running the aggregator. Only references (JAX arrays are immutable);
        # overwritten each extract(). The camera head is never invoked for
        # dumps, so _past_kvs_camera stays unallocated under PERSIST_SCENE.
        self._last_out_list: list[jnp.ndarray] | None = None
        self._last_images: jnp.ndarray | None = None
        self._last_patch_start_idx: jnp.ndarray | None = None
        self._last_rgb: jnp.ndarray | None = None

        # Streaming cache fields; reset() clears them at episode boundaries.
        self._past_kvs_padded: list[CacheEntry] | None = None
        self._last_scores: jnp.ndarray | None = None
        self._past_kvs_camera: list[CacheEntry] | None = None
        self._frame_idx: int = 0

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
        self._anchor_tokens = 5 + (_IMG_SIZE // _PATCH_SIZE) ** 2
        uniform = max(total_budget // self._agg_depth, self._anchor_tokens)
        self._cache_max = uniform + self._anchor_tokens
        self._max_budget = self._cache_max - self._anchor_tokens
        self._num_heads = self._aggregator.num_heads
        self._head_dim = self._aggregator.embed_dim // self._num_heads

    def _finalize_static_budget_override(self) -> None:
        """Default to fixed budgets and validate explicit static budgets."""
        if self._max_budget <= self._anchor_tokens:
            raise ValueError(
                f"total_budget={self._total_budget} leaves no room beyond "
                f"{self._anchor_tokens} anchor tokens per block"
            )
        if self._budgets_static_override is None:
            self._budgets_static_override = (self._max_budget,) * self._agg_depth
            return
        if len(self._budgets_static_override) != self._agg_depth:
            raise ValueError(
                f"budgets_static length {len(self._budgets_static_override)} "
                f"!= aggregator depth {self._agg_depth}"
            )
        invalid = [
            budget
            for budget in self._budgets_static_override
            if budget > self._max_budget or budget <= self._anchor_tokens
        ]
        if invalid:
            raise ValueError(
                "budgets_static entries must be in "
                f"({self._anchor_tokens}, {self._max_budget}], got {invalid}"
            )

    def _configure_camera_cache(self, max_camera_frames: int) -> None:
        """Derive padded cache dimensions for the fixed-window camera head."""
        self._cam_num_heads = self._camera_head.num_heads
        self._cam_head_dim = self._camera_head.dim_in // self._cam_num_heads
        self._cam_num_iters = self._camera_head.num_iterations
        self._cam_max = max_camera_frames * self._cam_num_iters

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
        batch = 1
        k_pad = jnp.zeros(
            (batch, self._num_heads, self._cache_max, self._head_dim),
            dtype=self._dtype,
        )
        v_pad = jnp.zeros(
            (batch, self._num_heads, self._cache_max, self._head_dim),
            dtype=self._dtype,
        )
        valid_len = jnp.asarray(0, dtype=jnp.int32)
        return (k_pad, v_pad, valid_len)

    def _new_padded_camera_entry(self) -> CacheEntry:
        """Allocate a zero-padded camera-head (k, v, valid_len=0) entry."""
        batch = 1
        k_pad = jnp.zeros(
            (batch, self._cam_num_heads, self._cam_max, self._cam_head_dim),
            dtype=self._dtype,
        )
        v_pad = jnp.zeros(
            (batch, self._cam_num_heads, self._cam_max, self._cam_head_dim),
            dtype=self._dtype,
        )
        valid_len = jnp.asarray(0, dtype=jnp.int32)
        return (k_pad, v_pad, valid_len)

    def _compute_static_budgets(self, last_scores_np: jnp.ndarray) -> tuple[int, ...]:
        """Compute dynamic per-block budgets as a tuple of Python ints."""
        bud = jnp.asarray(
            _calculate_dynamic_budgets(
                jnp.asarray(last_scores_np, dtype=jnp.float32),
                self._total_budget,
            )
        )
        # Leave room for the next frame append before eviction runs.
        bud = jnp.clip(bud, self._anchor_tokens + 1, self._max_budget)
        return tuple(int(x) for x in bud.tolist())

    def _warmup(self) -> None:
        """Pre-compile both cache states so the first real call is fast."""
        dummy = jnp.zeros((1, 1, 3, _IMG_SIZE, _IMG_SIZE), dtype=self._dtype)

        # Frame 0: padded cache with valid_len=0 on every block.
        past0 = [self._new_padded_cache_entry() for _ in range(self._agg_depth)]
        last0 = jnp.zeros((self._agg_depth,), dtype=jnp.float32)
        bud0 = self._compute_static_budgets(jnp.zeros(self._agg_depth, dtype=jnp.float32))
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
        out_list0, patch_start_idx0, past1, last1 = out0
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

        if not self._compute_heads:
            return

        # Camera-head warmup: single graph covers all frames since the padded
        # cache keeps shapes stable regardless of valid_len.
        past_cam0 = [self._new_padded_camera_entry() for _ in range(self._cam_depth)]
        pose_list_w, _ = self._camera_head_apply(self._cam_params, out_list0, past_cam0)
        pose_list_w[-1].block_until_ready()

        # Point-head warmup: unlike the aggregator and camera head, this was not
        # exercised above, so the first real extract() paid the DPT JIT cost.
        pts3d_w, _ = self._point_head_apply(
            self._pt_params,
            out_list0,
            dummy,
            int(jnp.asarray(patch_start_idx0)),
        )
        pts3d_w.block_until_ready()

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
            vl = int(jnp.asarray(valid_len))
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
        keys, values = entry
        batch, num_heads, seq_len, head_dim = keys.shape
        k_pad = jnp.zeros((batch, num_heads, self._cache_max, head_dim), dtype=keys.dtype)
        v_pad = jnp.zeros(
            (batch, num_heads, self._cache_max, head_dim), dtype=values.dtype
        )
        k_pad = jax.lax.dynamic_update_slice_in_dim(k_pad, keys, 0, axis=2)
        v_pad = jax.lax.dynamic_update_slice_in_dim(v_pad, values, 0, axis=2)
        valid_len = jnp.asarray(seq_len, dtype=jnp.int32)
        return (k_pad, v_pad, valid_len)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear KV-cache and frame counter. Call at episode boundaries."""
        self._past_kvs_padded = None
        self._last_scores = None
        self._past_kvs_camera = None
        self._frame_idx = 0

    def save_cache(self) -> AggregatorCacheSnapshot:
        """Snapshot the current streaming state (the four cache fields).

        Returns an :class:`AggregatorCacheSnapshot` holding references to the
        live arrays — no device-memory copy. Safe because JAX arrays are
        immutable and the extractor reassigns these fields each frame rather
        than mutating them. Use to persist the VGGT attention stream across
        episodes/agents of one scene (HANDOFF.md §4.3).
        """
        return AggregatorCacheSnapshot(
            past_kvs_padded=(
                tuple(self._past_kvs_padded) if self._past_kvs_padded is not None else None
            ),
            last_scores=self._last_scores,
            past_kvs_camera=(
                tuple(self._past_kvs_camera) if self._past_kvs_camera is not None else None
            ),
            frame_idx=self._frame_idx,
        )

    def load_cache(self, snapshot: AggregatorCacheSnapshot) -> None:
        """Restore streaming state from a snapshot produced by ``save_cache``.

        Restores the four fields verbatim. The caller is responsible for
        ensuring the snapshot came from a compatible extractor (same
        ``total_budget`` / ``max_camera_frames`` configuration), otherwise the
        padded cache shapes will not match the compiled apply functions.
        """
        self._past_kvs_padded = (
            list(snapshot.past_kvs_padded) if snapshot.past_kvs_padded is not None else None
        )
        self._last_scores = snapshot.last_scores
        self._past_kvs_camera = (
            list(snapshot.past_kvs_camera) if snapshot.past_kvs_camera is not None else None
        )
        self._frame_idx = snapshot.frame_idx

    def reset_for_scene(self, scene_id: str) -> None:
        """Mode-aware reset at an episode boundary.

        Under ``ResetMode.FULL`` (default) this is identical to :meth:`reset` —
        the streaming state is wiped every episode, matching the existing
        training behaviour.

        Under ``ResetMode.PERSIST_SCENE`` the current scene's state is saved
        (keyed by ``scene_id``) and the incoming scene's saved state is
        restored, or a fresh cache is started if the scene has not been seen
        before. Same-scene episodes thus resume the VGGT attention stream
        instead of re-anchoring. The camera-head cache is bounded by a sliding
        window (``_check_camera_cache_capacity`` -> ``_evict_oldest_camera_frame``,
        commit 6977127), so long per-scene streams no longer raise; the remaining
        cost is the per-frame eviction concat (HANDOFF.md R5) once
        ``max_camera_frames`` fills — watch the step-time regression vs the FULL
        baseline when enabling this for long runs.

        Args:
          scene_id: Identifier for the incoming scene. When it matches the
            current scene the call is a no-op.

        Returns:
          None.
        """
        if self._reset_mode is ResetMode.FULL:
            self.reset()
            return
        # PERSIST_SCENE: only save/restore when the scene actually changes.
        if self._current_scene_id == scene_id:
            return
        if self._current_scene_id is not None:
            self._scene_cache_store[self._current_scene_id] = self.save_cache()
        self._current_scene_id = scene_id
        saved = self._scene_cache_store.get(scene_id)
        if saved is not None:
            self.load_cache(saved)
        else:
            self.reset()

    def _image_from_extract_input(
        self, source: jnp.ndarray | ObservationFrame
    ) -> jnp.ndarray:
        """Resolve either a raw HWC frame or an ObservationFrame to image input."""
        if isinstance(source, ObservationFrame):
            if source.is_first:
                # Scene-aware episode reset. Under ``ResetMode.FULL`` this is
                # identical to ``self.reset()``. Under ``PERSIST_SCENE`` it
                # saves the outgoing scene's cache and restores the incoming
                # one. The reset lives here (not in the trainer's
                # ``on_episode_reset`` callback) because only the frame carries
                # the incoming ``scene_id`` the restore needs.
                self.reset_for_scene(
                    getattr(source, "scene_id", None) or "scene"
                )
            return source.image
        return source

    def _prepare_input_image(self, rgb: jnp.ndarray) -> jnp.ndarray:
        """Normalize an HWC uint8 frame and add batch/sequence dimensions.

        The observation contract is HWC; the aggregator/backbone plumbing (and
        the ported weights) stay NCHW, so the layout flip happens exactly once
        here.
        """
        if rgb.shape != (_IMG_SIZE, _IMG_SIZE, 3):
            raise ValueError(
                f"VGGT extractor expects HWC image shape "
                f"({_IMG_SIZE}, {_IMG_SIZE}, 3), got {rgb.shape}"
            )
        if rgb.dtype != jnp.uint8:
            raise ValueError(f"VGGT extractor expects uint8 image, got {rgb.dtype}")
        rgb_chw = jnp.transpose(rgb, (2, 0, 1))
        img = (jnp.asarray(rgb_chw, dtype=jnp.float32) / 255.0).astype(self._dtype)
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
        return self._compute_static_budgets(jnp.asarray(self._last_scores))

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
        """Evict the oldest camera frame when the padded cache is full.

        Sliding-window replacement for the former ``RuntimeError`` guard. The
        camera head has no internal eviction (its trunk blocks run with
        ``cache_budget=None``, unlike the aggregator), so without this the
        padded buffer would overflow: ``dynamic_update_slice_in_dim`` clamps
        out-of-range writes (silently corrupting the last slot) and
        ``key_value_seq_lengths > _cam_max`` gives undefined cuDNN behaviour.
        Instead of raising, drop one frame per new frame once full, so the
        camera head always attends over the most recent ``max_camera_frames``
        frames — matching the reference extractor's
        ``k[:, :, -max_camera_tokens:, :]`` sliding window
        (``reference/feature_extractor.py:214``). Output is byte-identical to
        the old behaviour for the first ``max_camera_frames`` frames; only
        behaviour past that point changes (continue instead of crash).
        """
        max_frames = self._cam_max // self._cam_num_iters
        if self._frame_idx >= max_frames:
            self._evict_oldest_camera_frame()

    def _evict_oldest_camera_frame(self) -> None:
        """Slide the camera-head cache left by one frame to make room for the next.

        Drops the oldest ``_cam_num_iters`` rows (one frame's worth, 4 pose
        iterations) from each trunk block and pads the same number of zero
        rows at the tail, setting ``valid_len = _cam_max - _cam_num_iters`` so
        the next camera-head apply appends the new frame at the freed end and
        ``valid_len`` returns to ``_cam_max``. Net: a sliding window of the
        last ``max_camera_frames`` frames. Run only when the cache is full.

        Cost (HANDOFF.md R5): once full, one concat per trunk block per frame
        (``_cam_depth`` blocks x 2 (k,v)). At ``_cam_max=4096`` this is
        non-trivial device traffic — watch the wall-clock regression vs. the
        158 ms/step baseline (EXPERIMENTS E2). A circular-buffer variant would
        avoid the copy but needs rotated reads inside the attention block.
        """
        if self._past_kvs_camera is None:
            return
        n = self._cam_num_iters
        shifted: list[CacheEntry] = []
        for k_pad, v_pad, valid_len in self._past_kvs_camera:
            vl = int(jnp.asarray(valid_len))
            if vl < self._cam_max:
                # Not full yet — no eviction; the apply will append and grow valid_len.
                shifted.append((k_pad, v_pad, valid_len))
                continue
            k_tail = jnp.zeros_like(k_pad[:, :, :n, :])
            v_tail = jnp.zeros_like(v_pad[:, :, :n, :])
            k_shift = jnp.concatenate([k_pad[:, :, n:, :], k_tail], axis=2)
            v_shift = jnp.concatenate([v_pad[:, :, n:, :], v_tail], axis=2)
            shifted.append(
                (k_shift, v_shift, jnp.asarray(self._cam_max - n, dtype=jnp.int32))
            )
        self._past_kvs_camera = shifted

    def _run_heads(
        self,
        out_list: list[jnp.ndarray],
        images: jnp.ndarray,
        patch_start_idx: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
        pts3d, conf = self._point_head_apply(
            self._pt_params,
            out_list,
            images,
            int(jnp.asarray(patch_start_idx)),
        )
        return pts3d[:, 0], conf[:, 0], camera_pose

    def _aggregator_full_tokens(self, out_list: list[jnp.ndarray]) -> jnp.ndarray:
        """Expose final full-width frame+global aggregator tokens."""
        final_tokens = out_list[-1]
        return final_tokens[0, 0].astype(jnp.float32)

    def _run_optional_heads(
        self,
        out_list: list[jnp.ndarray],
        images: jnp.ndarray,
        patch_start_idx: jnp.ndarray,
        *,
        phase_times: PhaseTimes | None,
        forward_start: float,
    ) -> HeadOutputs | None:
        """Run camera/point heads when enabled and record optional phase timings."""
        if not self._compute_heads:
            self._record_aggregator_only_profile(phase_times, forward_start)
            return None

        pts3d, conf, camera_pose = self._run_heads(out_list, images, patch_start_idx)
        wrapper_start = self._synchronize_heads_for_profile(
            pts3d,
            conf,
            camera_pose,
            phase_times,
        )
        head_outputs = HeadOutputs(pts3d, conf, camera_pose)
        self._record_head_profile(phase_times, forward_start, wrapper_start)
        return head_outputs

    def _synchronize_heads_for_profile(
        self,
        pts3d: jnp.ndarray,
        conf: jnp.ndarray,
        camera_pose: jnp.ndarray,
        phase_times: PhaseTimes | None,
    ) -> float:
        """Synchronize head tensors only when wall-clock profiling is active."""
        if phase_times is None:
            return 0.0
        pts3d.block_until_ready()
        conf.block_until_ready()
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
        frame_tokens: jnp.ndarray,
        global_tokens: jnp.ndarray,
        head_outputs: HeadOutputs | None,
    ) -> ExtractOutput:
        """Build the public structured output."""
        if self._compute_heads:
            if head_outputs is None:
                raise RuntimeError("head outputs missing while compute_heads=True")
            return VGGTExtractOutput(
                world_points=head_outputs.world_points,
                confidence=head_outputs.confidence,
                camera_pose=head_outputs.camera_pose,
                frame_tokens=frame_tokens,
                global_tokens=global_tokens,
            )
        # Heads disabled: no point map / confidence / camera pose for this frame.
        # The dataclass fields stay non-optional so head-enabled consumers need no
        # None guards.
        return VGGTExtractOutput(
            world_points=None,  # type: ignore[arg-type]
            confidence=None,  # type: ignore[arg-type]
            camera_pose=None,  # type: ignore[arg-type]
            frame_tokens=frame_tokens,
            global_tokens=global_tokens,
        )

    def extract(
        self,
        source: jnp.ndarray | ObservationFrame,
        phase_times: PhaseTimes | None = None,
        return_dense: bool = False,
    ) -> ExtractOutput:
        """Single-frame streaming inference from an ObservationFrame or HWC image.

        Passing an ``ObservationFrame`` is the preferred high-level path: the
        extractor resets its stream when ``source.is_first`` is true and then
        consumes ``source.image``. Passing a raw ``(518, 518, 3)`` uint8 array is
        still the low-level path for profiling and fixtures.

        The returned ``world_points`` field is the full 518x518x3 point map
        when heads are enabled. ``return_dense`` is accepted for call-site
        compatibility and no longer changes the output contract.
        """
        forward_start = time.perf_counter() if phase_times is not None else 0.0
        del return_dense

        rgb = self._image_from_extract_input(source)
        images = self._prepare_input_image(rgb)
        self._ensure_aggregator_cache()
        budgets_static = self._resolve_static_budgets()
        out_list, patch_start_idx = self._run_aggregator(images, budgets_static)

        # Retain this frame for an optional PLY snapshot (point head only).
        self._last_out_list = out_list
        self._last_images = images
        self._last_patch_start_idx = patch_start_idx
        self._last_rgb = rgb

        head_outputs = self._run_optional_heads(
            out_list,
            images,
            patch_start_idx,
            phase_times=phase_times,
            forward_start=forward_start,
        )
        aggregator_full_tokens = self._aggregator_full_tokens(out_list)
        token_split = aggregator_full_tokens.shape[-1] // 2
        frame_tokens = aggregator_full_tokens[..., :token_split]
        global_tokens = aggregator_full_tokens[..., token_split:]
        self._frame_idx += 1
        return self._build_extract_output(
            frame_tokens=frame_tokens,
            global_tokens=global_tokens,
            head_outputs=head_outputs,
        )

    def write_point_cloud_ply(self, path: str) -> None:
        """Write the last extracted frame's colored point cloud as a PLY.

        Runs the point head on the retained last-frame aggregator outputs, so
        it works regardless of ``compute_heads`` and adds no per-step cost.
        The camera head is never invoked, keeping its KV cache unallocated
        under ``PERSIST_SCENE``. Output is binary PLY via
        ``shared.ply_io.write_world_points_ply``; read it back with
        ``open3d.io.read_point_cloud``.

        Args:
          path: Output ``.ply`` path; parent directories are created.

        Raises:
          RuntimeError: If called before any extract().
        """
        if self._last_out_list is None or self._last_images is None:
            raise RuntimeError("write_point_cloud_ply called before any extract()")
        pts3d, _ = self._point_head_apply(
            self._pt_params,
            self._last_out_list,
            self._last_images,
            int(jnp.asarray(self._last_patch_start_idx)),
        )
        write_world_points_ply(path, pts3d[:, 0], self._last_rgb)
