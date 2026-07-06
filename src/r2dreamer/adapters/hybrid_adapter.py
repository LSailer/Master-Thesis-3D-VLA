"""Observation adapters that wrap a VGGT extractor for the house/hybrid encoders.

Hosts the :class:`ObsAdapter` subclasses that bridge a live VGGT extractor to
the CNN+WP/CP hybrid encoder and the L1 live-house-context encoders
(:class:`HybridObsAdapter`, :class:`VGGTHouseContextObsAdapter`,
:class:`VGGTHouseFullTokenObsAdapter`, :class:`VGGTHouseGlobalTokenObsAdapter`,
:class:`VGGTHouseGlobalEmbeddingObsAdapter`,
:class:`VGGTHousePointsPoseObsAdapter`).

The stateless collaborators these adapters compose now live in sibling modules
and are re-exported here (so existing ``hybrid_adapter`` import paths keep
working):

- :mod:`src.r2dreamer.adapters.scene_buffer` — ``SceneBufferManager``,
  ``default_house_context_pose_buffer_factory``, ``HouseContextPoseBufferLike``,
  ``BufferFactory``.
- :mod:`src.r2dreamer.adapters.subsampling` — ``InputSubsamplingPolicy`` and the
  VGGT output-field accessors.
- :mod:`src.r2dreamer.adapters.house_diagnostics` — ``HouseBufferDiagnostics``.
- :mod:`src.r2dreamer.adapters.point_cloud_dumper` — ``PointCloudDumper``,
  ``PointCloudDumpingExtractor``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from src.buffer.replay_buffer import ReplayBatch
from src.environments.observation import ObservationFrame
from src.r2dreamer.adapters.house_diagnostics import HouseBufferDiagnostics
from src.r2dreamer.adapters.obs_adapter import ObsAdapter
from src.r2dreamer.adapters.point_cloud_dumper import (
    PointCloudDumper,
    PointCloudDumpingExtractor,
)
from src.r2dreamer.adapters.scene_buffer import (
    BufferFactory,
    HouseContextPoseBufferLike,
    SceneBufferManager,
    default_house_context_pose_buffer_factory,
)
from src.r2dreamer.adapters.subsampling import (
    InputSubsamplingPolicy,
    _required_vggt_output_field,
    _vggt_output_field,
)
from src.r2dreamer.encoders.constants import (
    HOUSE_CONTEXT_DIM,
    HOUSE_CONTEXT_MAX_POINTS,
    HOUSE_POINT_DIM,
)
from src.r2dreamer.encoders.transformer import TokenTransformerEncoder
from src.r2dreamer.observation_keys import (
    CAMERA_POSE_KEY,
    CAMERA_TOKEN_GLOBAL_KEY,
    FULL_TOKENS_KEY,
    GLOBAL_PATCH_TOKENS_KEY,
    GLOBAL_TOKENS_KEY,
    HOUSE_CONTEXT_KEY,
    HOUSE_CONTEXT_SIZE_KEY,
    HYBRID_IMAGE_KEY,
    HYBRID_WP_CP_KEY,
)
from src.r2dreamer.observation_preparation.static_house_context import (
    encode_static_house_context,
    load_ascii_ply_xyzrgb,
)
from src.r2dreamer.observation_preparation.vggt import (
    HYBRID_IMAGE_SHAPE as IMAGE_SHAPE,
)
from src.r2dreamer.observation_preparation.vggt import (
    HYBRID_IMAGE_SIZE as IMAGE_SIZE,
)
from src.r2dreamer.observation_preparation.vggt import (
    HYBRID_RGB_DIM,
    VGGT_AGGREGATOR_EMBED_DIM,
    VGGT_AGGREGATOR_PATCH_START_IDX,
    VGGT_AGGREGATOR_TOKEN_COUNT,
    VGGT_FULL_TOKEN_EMBED_DIM,
    build_hybrid_contract,
    wp_cp_dim,
)
from src.r2dreamer.observation_preparation.vggt_readouts import (
    VGGTOutputLike,
    flatten_world_points_camera_pose,
    full_aggregator_tokens,
)
from src.shared.video_utils import resize_chw_uint8

HYBRID_FEATURE_DIM = HYBRID_RGB_DIM + wp_cp_dim()
HOUSE_CONTEXT_FEATURE_DIM = HYBRID_RGB_DIM + HOUSE_CONTEXT_DIM
FULL_TOKEN_SHAPE = (VGGT_AGGREGATOR_TOKEN_COUNT, VGGT_FULL_TOKEN_EMBED_DIM)
GLOBAL_TOKEN_SHAPE = (VGGT_AGGREGATOR_TOKEN_COUNT, VGGT_AGGREGATOR_EMBED_DIM)
# House-global-embedding L1 split of the global-half tokens (1374, 1024):
# camera token [0:1] and patch tokens [5:] (4 registers dropped).
CAMERA_TOKEN_GLOBAL_SHAPE = (1, VGGT_AGGREGATOR_EMBED_DIM)
GLOBAL_PATCH_TOKENS_SHAPE = (
    VGGT_AGGREGATOR_TOKEN_COUNT - VGGT_AGGREGATOR_PATCH_START_IDX,
    VGGT_AGGREGATOR_EMBED_DIM,
)
CAMERA_POSE_SHAPE = (9,)


class HybridObsAdapter(ObsAdapter):
    """Builds the hybrid replay fields ``{"image": rgb64, "wp_cp": wp_cp}``.

    The env renders 518x518 CHW uint8 (for VGGT). Each step we run VGGT once to
    obtain world_points + camera_pose, downsample the same frame to 64x64 for the
    CNN branch, and store both modalities under explicit replay keys. The agent
    still packs them into the legacy flat encoder input at the JAX boundary.
    """

    def __init__(
        self,
        extractor,
        *,
        env_render_resolution: int | None = None,
        encoder_module_cls=None,
        agent_overrides=None,
        design_notes: str = "",
    ):
        self.contract = build_hybrid_contract(
            extractor,
            env_render_resolution=env_render_resolution,
            encoder_module_cls=encoder_module_cls,
            agent_overrides=agent_overrides,
            design_notes=design_notes,
        )
        super().__init__(
            buffer_dtype=self.contract.replay_observation.buffer_dtype(),
            buffer_shape=self.contract.replay_observation.buffer_shape(),
            normalize_on_sample=self.contract.replay_observation.buffer_normalize(),
            agent_obs_shape=self.contract.encoder_input.shape,
            on_episode_reset=lambda scene_id="scene": extractor.reset_for_scene(scene_id),
        )
        self._extractor = extractor

    def transform(
        self, env_obs: ObservationFrame
    ) -> tuple[dict[str, np.ndarray], dict]:
        out = self._extractor.extract(env_obs)  # image is 518 CHW uint8
        wp_cp = flatten_world_points_camera_pose(out)  # jnp (4116,)
        img64 = resize_chw_uint8(env_obs.image, IMAGE_SIZE)
        replay = {
            HYBRID_IMAGE_KEY: img64,
            HYBRID_WP_CP_KEY: np.asarray(wp_cp, dtype=np.float32),
        }
        agent_obs = {
            HYBRID_IMAGE_KEY: img64,
            HYBRID_WP_CP_KEY: jnp.asarray(replay[HYBRID_WP_CP_KEY]),
            "is_first": env_obs.is_first,
        }
        return replay, agent_obs


class _RGBLiveTokenObsAdapter(ObsAdapter):
    token_key: str
    token_shape: tuple[int, int]
    token_name: str

    def __init__(self, extractor):
        super().__init__(
            buffer_dtype={HYBRID_IMAGE_KEY: "uint8", self.token_key: "float16"},
            buffer_shape={
                HYBRID_IMAGE_KEY: IMAGE_SHAPE,
                self.token_key: self.token_shape,
            },
            normalize_on_sample={HYBRID_IMAGE_KEY: True, self.token_key: False},
            agent_obs_shape={
                HYBRID_IMAGE_KEY: IMAGE_SHAPE,
                self.token_key: self.token_shape,
            },
            on_episode_reset=None,
        )
        self._extractor = extractor

    def _tokens_from_output(self, out: VGGTOutputLike) -> jnp.ndarray:
        raise NotImplementedError

    def _extract_tokens(self, obs: ObservationFrame) -> np.ndarray:
        tokens = self._tokens_from_output(self._extractor.extract(obs))
        if tuple(tokens.shape) != self.token_shape:
            raise ValueError(
                f"expected VGGT {self.token_name} shape {self.token_shape}, got {tokens.shape}"
            )
        return np.asarray(tokens, dtype=np.float32)

    def transform(
        self, env_obs: ObservationFrame
    ) -> tuple[dict[str, np.ndarray], dict]:
        image64 = resize_chw_uint8(env_obs.image, IMAGE_SIZE)
        tokens = self._extract_tokens(env_obs)
        replay = {
            HYBRID_IMAGE_KEY: image64,
            self.token_key: tokens.astype(np.float16),
        }
        return replay, {
            HYBRID_IMAGE_KEY: image64,
            self.token_key: jnp.asarray(tokens, dtype=jnp.float32),
            "is_first": env_obs.is_first,
        }


class VGGTHouseFullTokenObsAdapter(_RGBLiveTokenObsAdapter):
    """RGB replay plus live full VGGT tokens for the no-gate Transformer test."""

    token_key = FULL_TOKENS_KEY
    token_shape = FULL_TOKEN_SHAPE
    token_name = "full tokens"

    def _tokens_from_output(self, out: VGGTOutputLike) -> jnp.ndarray:
        return full_aggregator_tokens(out, self.token_shape)


class VGGTHouseGlobalTokenObsAdapter(_RGBLiveTokenObsAdapter):
    """RGB replay plus live singleton VGGT global-half tokens."""

    token_key = GLOBAL_TOKENS_KEY
    token_shape = GLOBAL_TOKEN_SHAPE
    token_name = "global tokens"

    def _tokens_from_output(self, out: VGGTOutputLike) -> jnp.ndarray:
        if isinstance(out, Mapping):
            tokens = out.get(GLOBAL_TOKENS_KEY, out.get("aggregator_features"))
            if tokens is None:
                raise ValueError("VGGT output field 'global_tokens' is required")
            return jnp.asarray(tokens)
        return out.global_tokens


class VGGTHouseGlobalEmbeddingObsAdapter(ObsAdapter):
    """RGB replay plus the two split VGGT global-half token fields (L1).

    Replay stores the 64x64 RGB frame plus two float16 token fields split from
    the extractor's ``global_tokens`` ``(1374, 1024)``: the camera token
    ``camera_token_global`` ``(1, 1024)`` — the Position Signal — and the patch
    tokens ``global_patch_tokens`` ``(1369, 1024)``, dropping the 4 register
    tokens (Darcet et al. arXiv:2309.16588). The PointNet reducer encoder pools
    only the patches and keeps the camera token on its own side branch.

    The extractor runs ``ResetMode.PERSIST_SCENE`` with heads off (the agent
    reads global tokens, not a point map); the scene-aware
    ``on_episode_reset`` fires ``reset_for_scene`` at every episode boundary —
    including the two prefill sites that discard the reset frame — so the VGGT
    attention stream persists across episodes of one house instead of
    re-anchoring. This is the prefill-orphaning fix from
    ``src/prototyp/live_house_context/PROTOCOL.md`` §2: without the callback,
    ``reset_for_scene`` would not fire during prefill and the first train
    episode would fresh-``reset()``, orphaning the prefill frame.

    Optional point-cloud PLY snapshots (diagnostics only): when
    ``pointcloud_dump_every > 0`` and ``pointcloud_dump_dir`` is set, the
    adapter triggers ``extractor.write_point_cloud_ply(path)`` every N env
    steps and once at the end of the first episode. The point head runs only
    on those dump steps (never for training); the camera head is never
    invoked, so the camera-head cache stays unallocated under PERSIST_SCENE
    (mitigating the unbounded-growth risk — see PROBLEMS.md).
    """

    def __init__(
        self,
        extractor,
        *,
        pointcloud_dump_every: int = 0,
        pointcloud_dump_dir: str | None = None,
    ):
        super().__init__(
            buffer_dtype={
                HYBRID_IMAGE_KEY: "uint8",
                CAMERA_TOKEN_GLOBAL_KEY: "float16",
                GLOBAL_PATCH_TOKENS_KEY: "float16",
            },
            buffer_shape={
                HYBRID_IMAGE_KEY: IMAGE_SHAPE,
                CAMERA_TOKEN_GLOBAL_KEY: CAMERA_TOKEN_GLOBAL_SHAPE,
                GLOBAL_PATCH_TOKENS_KEY: GLOBAL_PATCH_TOKENS_SHAPE,
            },
            normalize_on_sample={
                HYBRID_IMAGE_KEY: True,
                CAMERA_TOKEN_GLOBAL_KEY: False,
                GLOBAL_PATCH_TOKENS_KEY: False,
            },
            agent_obs_shape={
                HYBRID_IMAGE_KEY: IMAGE_SHAPE,
                CAMERA_TOKEN_GLOBAL_KEY: CAMERA_TOKEN_GLOBAL_SHAPE,
                GLOBAL_PATCH_TOKENS_KEY: GLOBAL_PATCH_TOKENS_SHAPE,
            },
            on_episode_reset=lambda scene_id="scene": extractor.reset_for_scene(
                scene_id
            ),
        )
        self._extractor = extractor
        self._pointcloud_dumper = PointCloudDumper(
            extractor,
            dump_every=pointcloud_dump_every,
            dump_dir=pointcloud_dump_dir,
        )

    @staticmethod
    def _global_tokens_from_output(out: VGGTOutputLike) -> jnp.ndarray:
        if isinstance(out, Mapping):
            tokens = out.get(GLOBAL_TOKENS_KEY, out.get("aggregator_features"))
            if tokens is None:
                raise ValueError("VGGT output field 'global_tokens' is required")
            return jnp.asarray(tokens)
        return out.global_tokens

    def _split_tokens(
        self, out: VGGTOutputLike
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return ``(camera_token (1, D), patch_tokens (1369, D))`` from one frame.

        Args:
          out: VGGT extractor output (object or mapping) carrying
            ``global_tokens`` (1374, 1024).

        Returns:
          ``(camera_token, patch_tokens)`` split at the camera/register boundary.

        Raises:
          ValueError: If the split shapes do not match the expected layout.
        """
        global_tokens = self._global_tokens_from_output(out)
        camera_token = global_tokens[0:1]
        patch_tokens = global_tokens[VGGT_AGGREGATOR_PATCH_START_IDX:]
        if tuple(camera_token.shape) != CAMERA_TOKEN_GLOBAL_SHAPE:
            raise ValueError(
                f"expected camera_token_global shape {CAMERA_TOKEN_GLOBAL_SHAPE}, "
                f"got {camera_token.shape}"
            )
        if tuple(patch_tokens.shape) != GLOBAL_PATCH_TOKENS_SHAPE:
            raise ValueError(
                f"expected global_patch_tokens shape {GLOBAL_PATCH_TOKENS_SHAPE}, "
                f"got {patch_tokens.shape}"
            )
        return camera_token, patch_tokens

    @property
    def _dump_enabled(self) -> bool:
        """Back-compat accessor for the composed dumper's ``enabled`` state."""
        return self._pointcloud_dumper.enabled

    def transform(
        self, env_obs: ObservationFrame
    ) -> tuple[dict[str, np.ndarray], dict]:
        """Extract, split, and store the global tokens; trigger PLY dumps.

        Args:
          env_obs: One environment observation frame.

        Returns:
          ``(replay, agent_obs)``: replay holds the 64x64 RGB and the two
          float16 token fields; agent_obs holds the same as float32 plus
          ``is_first``.
        """
        # End-of-first-episode snapshot: on the second is_first (start of
        # episode 2 = end of episode 1), dump the *previous* frame the
        # extractor still holds before extract() overwrites it / restores the
        # scene cache. Mirrors the design's "one snapshot when the first
        # episode ends".
        self._pointcloud_dumper.on_episode_start(bool(env_obs.is_first))

        image64 = resize_chw_uint8(env_obs.image, IMAGE_SIZE)
        out = self._extractor.extract(env_obs)
        camera_token, patch_tokens = self._split_tokens(out)

        replay = {
            HYBRID_IMAGE_KEY: image64,
            CAMERA_TOKEN_GLOBAL_KEY: np.asarray(camera_token, dtype=np.float16),
            GLOBAL_PATCH_TOKENS_KEY: np.asarray(patch_tokens, dtype=np.float16),
        }
        agent_obs = {
            HYBRID_IMAGE_KEY: image64,
            CAMERA_TOKEN_GLOBAL_KEY: jnp.asarray(camera_token, dtype=jnp.float32),
            GLOBAL_PATCH_TOKENS_KEY: jnp.asarray(patch_tokens, dtype=jnp.float32),
            "is_first": env_obs.is_first,
        }

        self._pointcloud_dumper.on_step()
        return replay, agent_obs

    def diagnostics(self) -> dict[str, float]:
        """Return PLY-dump and extractor-cache diagnostics for the run summary.

        Returns:
          A dict with ``dump_count``, ``env_steps``, and
          ``camera_head_cache_active`` (0 = unallocated, as required under
          PERSIST_SCENE with heads off).
        """
        stats = {
            "house_global_embedding/dump_count": float(
                self._pointcloud_dumper.dump_count
            ),
            "house_global_embedding/env_steps": float(
                self._pointcloud_dumper.env_steps
            ),
        }
        # Camera-head cache must stay unallocated under PERSIST_SCENE with
        # heads off (risk #2 in PROBLEMS.md): expose its state for the smoke.
        cache = getattr(self._extractor, "_past_kvs_camera", "missing")
        stats["house_global_embedding/camera_head_cache_active"] = (
            0.0 if cache is None else 1.0 if cache != "missing" else -1.0
        )
        return stats


class TokenContextEncoderLike(Protocol):
    """Structural interface the house-context adapter calls on its encoder.

    Matches the subset of the ``flax.linen.Module`` API that
    :class:`VGGTHouseContextObsAdapter` actually invokes, so any Flax module
    (or test double) exposing ``init``/``apply`` with this signature can be
    injected in place of :class:`~src.r2dreamer.encoders.transformer.TokenTransformerEncoder`.
    """

    def init(
        self, rng: jax.Array, tokens: jnp.ndarray, *, train: bool = False
    ) -> Any:
        """Initialize parameters for a batched ``(1, *tokens.shape)`` input."""
        ...

    def apply(self, params: Any, tokens: jnp.ndarray, *, train: bool = False) -> Any:
        """Apply the encoder to unbatched ``tokens``, returning the context."""
        ...


def default_context_transformer() -> TokenTransformerEncoder:
    """Build the default live house-context token Transformer.

    Returns:
      A :class:`TokenTransformerEncoder` configured for the L1 house-context
      readout (mean-pooled 1024-d context from full 1374x2048 VGGT tokens).
    """
    return TokenTransformerEncoder(
        embed_dim=HOUSE_CONTEXT_DIM,
        token_dim=VGGT_FULL_TOKEN_EMBED_DIM,
        num_tokens=VGGT_AGGREGATOR_TOKEN_COUNT,
        model_dim=None,
        readout="mean",
        norm_kind="layer",
        activation="gelu",
    )


class VGGTHouseContextObsAdapter(ObsAdapter):
    """RGB replay plus live InfiniteVGGT house context for L1 experiments.

    Replay stores only the 64x64 RGB frame. The VGGT extractor remains live
    across episode resets; its bounded streaming cache supplies full
    1374x2048 tokens. A context Transformer maps those tokens to a cached
    1024-d context that is injected into sampled replay windows.
    """

    def __init__(
        self,
        extractor,
        *,
        context_transformer: TokenContextEncoderLike | None = None,
        rng_seed: int = 0,
        static_house_context_path: str | None = None,
        static_house_context: np.ndarray | None = None,
    ):
        if static_house_context_path is not None and static_house_context is not None:
            raise ValueError(
                "set only one of static_house_context_path or static_house_context"
            )
        super().__init__(
            buffer_dtype={HYBRID_IMAGE_KEY: "uint8", HOUSE_CONTEXT_KEY: "float16"},
            buffer_shape={
                HYBRID_IMAGE_KEY: IMAGE_SHAPE,
                HOUSE_CONTEXT_KEY: (HOUSE_CONTEXT_DIM,),
            },
            normalize_on_sample={HYBRID_IMAGE_KEY: True, HOUSE_CONTEXT_KEY: False},
            on_episode_reset=None,
        )
        self._extractor = extractor
        self._static_context = self._resolve_static_context(
            static_house_context_path,
            static_house_context,
        )
        self._context_transformer = context_transformer or default_context_transformer()
        self._context_params = None
        self._rng = jax.random.PRNGKey(rng_seed)
        self._context: np.ndarray | None = None
        self.agent_obs_shape = (HOUSE_CONTEXT_FEATURE_DIM,)

    @staticmethod
    def _resolve_static_context(
        path: str | None,
        context: np.ndarray | None,
    ) -> np.ndarray | None:
        if context is not None:
            return np.asarray(context, dtype=np.float16)
        if path is not None:
            return encode_static_house_context(load_ascii_ply_xyzrgb(path))
        return None

    def _ensure_context_params(self, tokens: jnp.ndarray):
        if self._context_params is None:
            self._context_params = self._context_transformer.init(
                self._rng,
                jnp.zeros((1, *tokens.shape), dtype=jnp.float32),
                train=False,
            )
        return self._context_params

    def _project_context(self, tokens: jnp.ndarray) -> np.ndarray:
        params = self._ensure_context_params(tokens)
        context = self._context_transformer.apply(params, tokens, train=False)
        context = np.asarray(context, dtype=np.float32)
        if context.shape != (HOUSE_CONTEXT_DIM,):
            raise ValueError(
                f"expected VGGT context shape {(HOUSE_CONTEXT_DIM,)}, got {context.shape}"
            )
        return context

    def _extract_context(self, obs: ObservationFrame) -> np.ndarray:
        if self._static_context is not None:
            self._context = self._static_context
            return self._static_context
        out = self._extractor.extract(obs)
        tokens = full_aggregator_tokens(out, FULL_TOKEN_SHAPE)
        context = self._project_context(tokens)
        self._context = context
        return context

    def transform(
        self, env_obs: ObservationFrame
    ) -> tuple[dict[str, np.ndarray], dict]:
        image64 = resize_chw_uint8(env_obs.image, IMAGE_SIZE)
        context = self._extract_context(env_obs)
        replay = {
            HYBRID_IMAGE_KEY: image64,
            HOUSE_CONTEXT_KEY: context.astype(np.float16),
        }
        agent_obs = {
            HYBRID_IMAGE_KEY: image64,
            HOUSE_CONTEXT_KEY: jnp.asarray(context, dtype=jnp.float32),
            "is_first": env_obs.is_first,
        }
        return replay, agent_obs


def _normalise_house_points(points_xyzrgb: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xyzrgb, dtype=np.float32).copy()
    rgb = points[:, 3:6]
    if float(np.max(rgb)) > 1.0:
        rgb = rgb / 255.0
    points[:, 3:6] = np.clip(rgb, 0.0, 1.0)
    return points.astype(np.float16)


def _camera_pose_from_output(out: VGGTOutputLike) -> np.ndarray:
    camera_pose = _vggt_output_field(out, "camera_pose")
    if camera_pose is None:
        raise ValueError("VGGT output field 'camera_pose' is required")
    return np.asarray(camera_pose, dtype=np.float16).reshape(CAMERA_POSE_SHAPE)


class VGGTHousePointsPoseObsAdapter(ObsAdapter):
    """Replay current camera pose plus a live, per-scene house point cloud.

    Replay stores only ``camera_pose`` per step. House-context points are
    accumulated live from VGGT world points into one
    :class:`HouseContextPoseBuffer` per ``scene_id`` (mirroring
    ``ScenePointCloudTracker.point_clouds`` in the ``live_vggt`` prototype).
    Every step the adapter emits a fixed-size
    ``(max_points, 6)`` snapshot zero-padded from the growing buffer, plus the
    true valid-row count under ``HOUSE_CONTEXT_SIZE_KEY`` for masked pooling in
    the encoder, so ``jax.jit`` sees a stable house-context shape and never
    recompiles ``train_step``/``act`` as the cloud grows. Only if the buffer
    outgrows ``max_points`` does the snapshot fall back to an even-stride
    subsample.

    An optional ``house_points_path`` warm-starts every new scene buffer from a
    static ASCII XYZRGB PLY; live VGGT points then extend it. The extractor is
    constructed with ``ResetMode.PERSIST_SCENE`` and resets scene-aware inside
    ``extract`` on the first frame of each episode (``reset_for_scene``), so its
    streaming KV-cache is saved/restored per scene rather than wiped — keeping
    every episode of one house in a single world frame — while the per-scene
    point buffers persist across episodes of the same scene.

    The full VGGT point map (518x518 ~ 268k points) is fed to the buffer every
    step by default: voxel dedup runs as a fixed-shape JIT graph on device
    (~2 ms/frame on H100 regardless of stored size), so no subsampling is
    needed. Passing ``max_input_points > 0`` restores an even-stride subsample
    as an opt-in bound for constrained setups.

    Note (single-scene contract): ``augment_replay_batch`` injects the latest
    scene's snapshot, and ``_house_embedding`` broadcasts that one cloud across
    the whole camera batch — matching the model's global-house-context design
    and the previous static-sidecar behaviour. For multi-scene training this is
    an approximation (the sampled camera poses may come from another scene); the
    L1 curriculum is single-house so it is exact there.
    """

    DEFAULT_CONFIDENCE_SCORE = 1.5
    DEFAULT_VOXEL_SIZE_M = 0.01
    DEFAULT_MAX_INPUT_POINTS = 0
    # 1 cm voxels on full-res input reached ~210k points in 50 steps on one L1
    # scene; a whole house is bounded by surface area (~500 m^2 -> ~5M voxels),
    # so 2^23 (~293 MB device memory) covers any realistic scene outright.
    BUFFER_CAPACITY = 1 << 23
    BUFFER_HASH_TABLE_SIZE = 1 << 24

    def __init__(
        self,
        extractor,
        *,
        house_points_path: str | None = None,
        confidence_score: float = DEFAULT_CONFIDENCE_SCORE,
        voxel_size_m: float = DEFAULT_VOXEL_SIZE_M,
        max_points: int = HOUSE_CONTEXT_MAX_POINTS,
        max_input_points: int = DEFAULT_MAX_INPUT_POINTS,
        buffer_factory: BufferFactory | None = None,
    ):
        self._confidence_score = float(confidence_score)
        self._voxel_size_m = float(voxel_size_m)
        self._max_points = int(max_points)
        self._max_input_points = int(max_input_points)
        seed_xyzrgb = self._load_seed(house_points_path)
        factory = buffer_factory or default_house_context_pose_buffer_factory(
            confidence_score=self._confidence_score,
            voxel_size_m=self._voxel_size_m,
            capacity=self.BUFFER_CAPACITY,
            hash_table_size=self.BUFFER_HASH_TABLE_SIZE,
        )
        self._scene_buffers = SceneBufferManager(factory, seed_xyzrgb=seed_xyzrgb)
        self._subsampler = InputSubsamplingPolicy(self._max_input_points)
        self._buffer_diagnostics = HouseBufferDiagnostics(self._scene_buffers)
        self._latest_house_context = self._empty_house_context()
        self._latest_house_context_size = jnp.zeros((), dtype=jnp.int32)
        super().__init__(
            buffer_dtype={CAMERA_POSE_KEY: "float16"},
            buffer_shape={CAMERA_POSE_KEY: CAMERA_POSE_SHAPE},
            normalize_on_sample={CAMERA_POSE_KEY: False},
            agent_obs_shape={
                CAMERA_POSE_KEY: CAMERA_POSE_SHAPE,
                HOUSE_CONTEXT_KEY: (self._max_points, HOUSE_POINT_DIM),
                HOUSE_CONTEXT_SIZE_KEY: (),
            },
            on_episode_reset=lambda scene_id="scene": extractor.reset_for_scene(scene_id),
        )
        # Scene-aware episode reset. The trainer calls the callback above at
        # every episode boundary — including the two prefill sites that discard
        # the reset frame (so the in-extract ``is_first`` reset path never fires
        # during prefill). Without this callback, ``reset_for_scene`` would not
        # run during prefill, ``_current_scene_id`` would stay None, and the
        # first train episode would fresh-``reset()`` (re-anchor), orphaning the
        # prefill frame — the bug found in smoke 5738008 (3.77 M points, no
        # saturation). See src/prototyp/live_house_context/PROTOCOL.md §2.
        # The earlier ``on_episode_reset=None`` was set to avoid a FULL-wipe
        # hazard, but that hazard only applied to a bare ``extractor.reset``
        # callback; ``reset_for_scene`` saves the outgoing scene before
        # restoring the incoming one, so there is no wipe. The in-extract
        # ``is_first`` -> ``reset_for_scene`` (set in feature_extractor.py) stays
        # as a redundant, idempotent safety net for paths that process the reset
        # frame (the train loop). Only the point buffer persists across
        # episodes; the VGGT cache is saved/restored per scene.
        self._extractor = extractor

    @staticmethod
    def _load_seed(house_points_path: str | None) -> jnp.ndarray | None:
        """Load an optional static PLY as an ``(M, 6)`` [0, 1] warm-start seed."""
        if house_points_path is None:
            return None
        seed = _normalise_house_points(load_ascii_ply_xyzrgb(house_points_path))
        return jnp.asarray(seed, dtype=jnp.float32)

    def _empty_house_context(self) -> jnp.ndarray:
        """Return the fixed-shape all-zeros house context used before any add."""
        return jnp.zeros((self._max_points, HOUSE_POINT_DIM), dtype=jnp.float16)

    @property
    def _buffers(self) -> Mapping[str, HouseContextPoseBufferLike]:
        """Back-compat view of the live per-scene buffers (tests, scripts)."""
        return self._scene_buffers.buffers

    def _get_or_create_buffer(self, scene_id: str) -> HouseContextPoseBufferLike:
        """Return the buffer for ``scene_id``, creating and seeding it once."""
        return self._scene_buffers.get_or_create(scene_id)

    def _house_context_snapshot(
        self, buffer: HouseContextPoseBufferLike
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return a JIT-stable ``((max_points, 6) float16, () int32)`` snapshot.

        The scalar is the valid-row count; rows beyond it are zero padding the
        encoder masks out during pooling.
        """
        return buffer.house_context_array(self._max_points, dtype=jnp.float16)

    def _input_stride(self, height: int, width: int) -> int:
        """Return the even stride that caps a ``(height, width)`` map to ~budget."""
        return self._subsampler.stride(height, width)

    def _subsampled_buffer_input(
        self, out: VGGTOutputLike, env_obs: ObservationFrame
    ) -> tuple[Any, ObservationFrame]:
        """Return ``(vggt_output, observation)`` strided to bound ``add`` cost."""
        return self._subsampler.subsample(out, env_obs)

    def transform(
        self, env_obs: ObservationFrame
    ) -> tuple[dict[str, np.ndarray], dict]:
        out = self._extractor.extract(env_obs)
        camera_pose = _camera_pose_from_output(out)
        buffer = self._scene_buffers.get_or_create(env_obs.scene_id)
        buffer_out, buffer_obs = self._subsampler.subsample(out, env_obs)
        buffer.add(buffer_out, buffer_obs)
        self._buffer_diagnostics.record_step()
        house_context, house_size = self._house_context_snapshot(buffer)
        self._latest_house_context = house_context
        self._latest_house_context_size = house_size
        replay = {CAMERA_POSE_KEY: camera_pose}
        agent_obs = {
            CAMERA_POSE_KEY: jnp.asarray(camera_pose, dtype=jnp.float16),
            HOUSE_CONTEXT_KEY: house_context,
            HOUSE_CONTEXT_SIZE_KEY: house_size,
            "is_first": env_obs.is_first,
        }
        return replay, agent_obs

    @property
    def growth_history(self) -> list[tuple[int, int]]:
        """``(env_step, total_points)`` samples at doubling env steps."""
        return self._buffer_diagnostics.growth_history

    def diagnostics(self) -> dict[str, float]:
        """Per-scene house-buffer usage; syncs one scalar per buffer to host."""
        return self._buffer_diagnostics.diagnostics()

    def augment_replay_batch(self, batch: ReplayBatch) -> ReplayBatch:
        """Inject the latest live house-context/pose into a sampled batch.

        Args:
            batch: Sampled replay batch (as returned by ``ReplayBuffer.sample``).

        Returns:
            The batch with ``HOUSE_CONTEXT_KEY`` and ``HOUSE_CONTEXT_SIZE_KEY``
            added to its observation mapping.
        """
        obs = dict(batch.obs)
        obs[HOUSE_CONTEXT_KEY] = self._latest_house_context
        obs[HOUSE_CONTEXT_SIZE_KEY] = self._latest_house_context_size
        return dataclasses.replace(batch, obs=obs)
