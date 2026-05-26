"""JAX StreamVGGT feature extractor for streaming 3D point extraction.

Mirrors the public API of ``src.vggt.feature_extractor.VGGTFeatureExtractor``
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


def _adaptive_avg_pool_518_to_37(pts_nhwc: jnp.ndarray) -> jnp.ndarray:
    """Average-pool (N, 518, 518, C) to (N, 37, 37, C)."""
    N, H, W, C = pts_nhwc.shape
    if (H, W) != (_IMG_SIZE, _IMG_SIZE):
        raise ValueError(f"expected (518, 518), got ({H}, {W})")
    out = pts_nhwc.reshape(
        N, _PATCH_GRID, _PATCH_SIZE, _PATCH_GRID, _PATCH_SIZE, C
    ).mean(axis=(2, 4))
    return out  # (N, 37, 37, C)


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
    ):
        # compute_heads=False skips camera_head + point_head + world_points
        # wrapper in extract(); only `aggregator_features` is returned. Used by
        # encoders that consume the pre-head aggregator tokens directly.
        self._compute_heads = compute_heads
        self._budgets_static_override = budgets_static
        if device in ("cuda", "gpu"):
            self._device = jax.devices("gpu")[0]
        elif device == "cpu":
            self._device = jax.devices("cpu")[0]
        else:
            raise ValueError(f"unknown device {device!r}")
        self._compile = compile  # reserved
        self._total_budget = total_budget
        self._dtype = dtype

        # Load weights: HF checkpoint -> numpy state_dict -> Flax PyTree.
        sd = load_checkpoint()
        tree, _ = load_pytorch_weights(sd, include_v1_only=True)
        tree = jax.tree.map(lambda x: jax.device_put(jnp.asarray(x), self._device), tree)
        self._agg_params = {"params": tree["aggregator"]}
        self._cam_params = {"params": tree["camera_head"]}
        self._pt_params = {"params": tree["point_head"]}

        self._aggregator = Aggregator()
        self._camera_head = CameraHead()
        self._point_head = DPTHead()

        self._agg_depth = self._aggregator.depth
        self._cam_depth = self._camera_head.trunk_depth

        # Padded cache dimensions.
        # Token count per global-block frame: P = 5 + (518/14)^2 = 1374.
        self._P = 5 + (_IMG_SIZE // _PATCH_SIZE) ** 2
        # MAX = uniform share of total_budget across depth blocks + one frame
        # headroom. After eviction the cache has <= uniform valid tokens, and
        # we append at most P tokens before the next eviction fires, so
        # uniform + P is the tightest safe upper bound (was uniform * 2).
        uniform = max(total_budget // self._agg_depth, self._P)
        self._MAX = uniform + self._P  # headroom for one fresh frame before eviction
        # Heads / head-dim from aggregator defaults.
        self._num_heads = self._aggregator.num_heads
        self._head_dim = self._aggregator.embed_dim // self._num_heads

        # Camera-head padded cache dims.  The camera head has its own
        # num_heads/dim_in (16 × 2048) and no eviction: each frame appends
        # num_iterations tokens to every trunk block's cache.  _CAM_MAX
        # therefore bounds the worst-case episode length.
        self._cam_num_heads = self._camera_head.num_heads
        self._cam_head_dim = self._camera_head.dim_in // self._cam_num_heads
        self._cam_num_iters = self._camera_head.num_iterations
        self._CAM_MAX = max_camera_frames * self._cam_num_iters

        # Point head JIT (unchanged from previous version).
        def _point_head_fn(params, out_list, images, psi):
            return self._point_head.apply(params, out_list, images, psi)

        self._point_head_apply = jax.jit(_point_head_fn, static_argnums=(3,))

        # JIT-wrap the aggregator apply. ``past_frame_idx`` (argnum 3) and
        # ``current_budgets_static`` (argnum 7) are Python-static.
        # Positional signature: (params, images, past_kvs, past_frame_idx,
        # total_budget, last_scores, use_cache, current_budgets_static).
        # is_first_frame (argnum 3) is a bool — only two distinct static values
        # → two compiles total. Previously used frame_idx (int), which caused
        # a recompile every frame.
        def _agg_fn(
            params,
            images,
            past_kvs,
            is_first_frame,
            total_budget,
            last_scores,
            use_cache,
            current_budgets_static,
        ):
            return self._aggregator.apply(
                params,
                images,
                use_cache=use_cache,
                past_kvs=past_kvs,
                past_frame_idx=0 if is_first_frame else 1,
                total_budget=total_budget,
                last_scores=last_scores,
                current_budgets_static=current_budgets_static,
            )

        # static_argnums: is_first_frame (3), total_budget (4), use_cache (6),
        # current_budgets_static (7).
        self._aggregator_apply = jax.jit(
            _agg_fn, static_argnums=(3, 4, 6, 7)
        )

        # JIT-wrap camera-head apply.  Past KVs are padded 3-tuples with
        # fixed MAX → graph is shape-stable across frames, no static args
        # required (a single compile covers frame 0 and frame N alike).
        def _cam_fn(params, out_list, past_kvs_camera):
            return self._camera_head.apply(
                params,
                out_list,
                use_cache=True,
                past_kvs_camera=past_kvs_camera,
            )

        self._camera_head_apply = jax.jit(_cam_fn)

        self.reset()

        # AOT warmup: compile for frame 0 (past_kvs=None → padded init with
        # valid_len=0) and frame 1 (past_kvs=3-tuples with valid_len>0).
        self._warmup()

    @property
    def patch_grid(self) -> int:
        """Number of VGGT patch tokens per spatial side for the configured image size."""
        return _IMG_SIZE // _PATCH_SIZE

    @property
    def aggregator_feature_shape(self) -> tuple[int, int, int]:
        """Shape of one frame's all-token pre-head global aggregator features."""
        return (1 + self._aggregator.num_register_tokens + self.patch_grid ** 2, self._aggregator.embed_dim)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _new_padded_cache_entry(self) -> tuple:
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

    def _new_padded_camera_entry(self) -> tuple:
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
        # Cap by MAX so top_k k values stay <= MAX - P.
        bud = np.minimum(bud, self._MAX)
        return tuple(int(x) for x in bud.tolist())

    def _warmup(self) -> None:
        """Pre-compile both cache states so the first real call is fast."""
        dummy = jnp.zeros(
            (1, 1, 3, _IMG_SIZE, _IMG_SIZE), dtype=self._dtype
        )

        # Frame 0: padded cache with valid_len=0 on every block.
        past0 = [self._new_padded_cache_entry() for _ in range(self._agg_depth)]
        last0 = jnp.zeros((self._agg_depth,), dtype=jnp.float32)
        bud0 = self._compute_static_budgets(np.zeros(self._agg_depth, dtype=np.float32))
        out0 = self._aggregator_apply(
            self._agg_params, dummy, past0, True, self._total_budget,
            last0, True, bud0,
        )
        out0[0][-1].block_until_ready()

        # Frame 1: reuse returned cache to compile the "is_first_frame=False" graph.
        _, _, past1, last1 = out0
        # Keep same budget (same last_scores pre-eviction) so shapes match.
        out1 = self._aggregator_apply(
            self._agg_params, dummy, past1, False, self._total_budget,
            last1, True, bud0,
        )
        out1[0][-1].block_until_ready()

        # Camera-head warmup: single graph covers all frames since the padded
        # cache keeps shapes stable regardless of valid_len.
        past_cam0 = [
            self._new_padded_camera_entry() for _ in range(self._cam_depth)
        ]
        pose_list_w, _ = self._camera_head_apply(
            self._cam_params, out0[0], past_cam0
        )
        pose_list_w[-1].block_until_ready()

    # ------------------------------------------------------------------
    # Public cache view (_past_kvs property).
    # ------------------------------------------------------------------

    @property
    def _past_kvs(self):
        """Compact 2-tuple view of the padded cache, for test observation.

        Returns None if no frames have been processed. Otherwise a list of
        ``(k, v)`` tuples, each shape ``(B, H, valid_len, Dh)``.
        """
        if self._past_kvs_padded is None:
            return None
        out = []
        for entry in self._past_kvs_padded:
            k_pad, v_pad, valid_len = entry
            vl = int(np.asarray(valid_len))
            out.append((k_pad[:, :, :vl], v_pad[:, :, :vl]))
        return out

    @_past_kvs.setter
    def _past_kvs(self, value):
        """Setter supporting the ``self._past_kvs = None`` reset pattern.

        Writing anything non-None falls through to storing as padded state.
        In practice this is only used by ``reset()``.
        """
        if value is None:
            self._past_kvs_padded = None
        else:
            # Caller providing either 2-tuples or 3-tuples; re-pad as needed.
            self._past_kvs_padded = [self._to_padded(e) for e in value]

    def _to_padded(self, entry) -> tuple:
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
        self._past_kvs_padded: list[Any] | None = None
        self._last_scores: jnp.ndarray | None = None
        self._past_kvs_camera: list[Any] | None = None
        self._frame_idx: int = 0

    def extract(
        self,
        rgb: np.ndarray,
        phase_times: dict[str, list[float]] | None = None,
    ) -> dict[str, jnp.ndarray]:
        """Single-frame streaming inference."""
        profiling = phase_times is not None
        fwd_t0 = time.perf_counter() if profiling else 0.0

        img = (jnp.asarray(rgb, dtype=jnp.float32) / 255.0).astype(self._dtype)
        images = img[None, None]
        images = jax.device_put(images, self._device)

        # Initialize padded cache on first frame.
        if self._past_kvs_padded is None:
            self._past_kvs_padded = [
                self._new_padded_cache_entry() for _ in range(self._agg_depth)
            ]
        if self._last_scores is None:
            self._last_scores = jnp.zeros((self._agg_depth,), dtype=jnp.float32)

        # Compute budgets outside jit as a static tuple of Python ints.
        if self._budgets_static_override is not None:
            budgets_static = self._budgets_static_override
        else:
            ls_np = np.asarray(self._last_scores)
            budgets_static = self._compute_static_budgets(ls_np)

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

        if self._compute_heads:
            if self._past_kvs_camera is None:
                self._past_kvs_camera = [
                    self._new_padded_camera_entry() for _ in range(self._cam_depth)
                ]
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
            pose_list, self._past_kvs_camera = self._camera_head_apply(
                self._cam_params,
                out_list,
                self._past_kvs_camera,
            )
            pose_enc = pose_list[-1]
            camera_pose = pose_enc[:, 0, :]

            # patch_start_idx is always 1 + num_register_tokens = 5; cast to Python
            # int so it can be used as a static JIT arg (JIT returns JAX scalars).
            pts3d, _ = self._point_head_apply(
                self._pt_params,
                out_list,
                images,
                int(np.asarray(patch_start_idx)),
            )
            pts3d = pts3d[:, 0]

            if profiling:
                pts3d.block_until_ready()
                camera_pose.block_until_ready()
                wrap_t0 = time.perf_counter()

            world_points = _adaptive_avg_pool_518_to_37(pts3d)

            world_points_out = world_points[0].astype(jnp.float32)
            camera_pose_out = camera_pose[0].astype(jnp.float32)

            if profiling:
                wrap_t1 = time.perf_counter()
                phase_times["vggt_forward"].append((wrap_t0 - fwd_t0) * 1000.0)
                phase_times["vggt_wrapper"].append((wrap_t1 - wrap_t0) * 1000.0)
        elif profiling:
            # No heads: forward timing ends right after aggregator, no wrapper work.
            wrap_t0 = time.perf_counter()
            phase_times["vggt_forward"].append((wrap_t0 - fwd_t0) * 1000.0)
            phase_times["vggt_wrapper"].append(0.0)

        # Final pre-head aggregator tokens for encoder ablations. The JAX port
        # stores frame/local and global/contextual streams concatenated as
        # 2048-d tokens for DPT heads. Expose them as two 1024-d halves so a
        # single forward feeds both readouts (3D-47): the global stream alone
        # (3072-d pooled) or frame ⊕ global together (6144-d pooled). Keep all
        # VGGT-DP / VGGT-World tokens: camera + register + spatial patches.
        final_tokens = out_list[-1]
        half = final_tokens.shape[-1] // 2
        final_global = final_tokens[..., half:]
        final_frame = final_tokens[..., :half]
        aggregator_features = final_global[0, 0].astype(jnp.float32)
        aggregator_features_frame = final_frame[0, 0].astype(jnp.float32)

        self._frame_idx += 1

        if self._compute_heads:
            return {
                "world_points": world_points_out,
                "camera_pose": camera_pose_out,
                "aggregator_features": aggregator_features,
                "aggregator_features_frame": aggregator_features_frame,
            }
        return {
            "aggregator_features": aggregator_features,
            "aggregator_features_frame": aggregator_features_frame,
        }
