"""HybridObsAdapter: wraps a VGGT extractor for the CNN+WP/CP hybrid encoder."""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
import jax
import numpy as np

from src.environments.observation import ObservationFrame
from src.r2dreamer.adapters.obs_adapter import ObsAdapter
from src.r2dreamer.observation_preparation.vggt_readouts import (
    VGGTOutputLike,
    flatten_world_points_camera_pose,
    full_aggregator_tokens,
)
from src.r2dreamer.obs_batch import (
    FULL_TOKENS_KEY,
    GLOBAL_TOKENS_KEY,
    HOUSE_CONTEXT_KEY,
    HYBRID_IMAGE_KEY,
    HYBRID_WP_CP_KEY,
)
from src.r2dreamer.observation_preparation.vggt import (
    HYBRID_IMAGE_SHAPE as IMAGE_SHAPE,
    HYBRID_IMAGE_SIZE as IMAGE_SIZE,
    HYBRID_RGB_DIM,
    VGGT_AGGREGATOR_EMBED_DIM,
    VGGT_AGGREGATOR_TOKEN_COUNT,
    VGGT_FULL_TOKEN_EMBED_DIM,
    build_hybrid_contract,
    wp_cp_dim,
)
from src.shared.video_utils import resize_chw_uint8
from src.r2dreamer.encoders.constants import HOUSE_CONTEXT_DIM
from src.r2dreamer.encoders.transformer import TokenTransformerEncoder


HYBRID_FEATURE_DIM = HYBRID_RGB_DIM + wp_cp_dim()
HOUSE_CONTEXT_FEATURE_DIM = HYBRID_RGB_DIM + HOUSE_CONTEXT_DIM
FULL_TOKEN_SHAPE = (VGGT_AGGREGATOR_TOKEN_COUNT, VGGT_FULL_TOKEN_EMBED_DIM)
GLOBAL_TOKEN_SHAPE = (VGGT_AGGREGATOR_TOKEN_COUNT, VGGT_AGGREGATOR_EMBED_DIM)


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
            on_episode_reset=extractor.reset,
        )
        self._extractor = extractor

    def transform(self, env_obs: ObservationFrame) -> tuple[dict[str, np.ndarray], dict]:
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
            buffer_shape={HYBRID_IMAGE_KEY: IMAGE_SHAPE, self.token_key: self.token_shape},
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

    def transform(self, env_obs: ObservationFrame) -> tuple[dict[str, np.ndarray], dict]:
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
        context_transformer: TokenTransformerEncoder | None = None,
        rng_seed: int = 0,
    ):
        super().__init__(
            buffer_dtype={HYBRID_IMAGE_KEY: "uint8", HOUSE_CONTEXT_KEY: "float32"},
            buffer_shape={
                HYBRID_IMAGE_KEY: IMAGE_SHAPE,
                HOUSE_CONTEXT_KEY: (HOUSE_CONTEXT_DIM,),
            },
            normalize_on_sample={HYBRID_IMAGE_KEY: True, HOUSE_CONTEXT_KEY: False},
            on_episode_reset=None,
        )
        self._extractor = extractor
        self._context_transformer = context_transformer or TokenTransformerEncoder(
            embed_dim=HOUSE_CONTEXT_DIM,
            token_dim=VGGT_FULL_TOKEN_EMBED_DIM,
            num_tokens=VGGT_AGGREGATOR_TOKEN_COUNT,
            model_dim=None,
            readout="mean",
            norm_kind="layer",
            activation="gelu",
        )
        self._context_params = None
        self._rng = jax.random.PRNGKey(rng_seed)
        self._context: np.ndarray | None = None
        self.agent_obs_shape = (HOUSE_CONTEXT_FEATURE_DIM,)

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
        out = self._extractor.extract(obs)
        tokens = full_aggregator_tokens(out, FULL_TOKEN_SHAPE)
        context = self._project_context(tokens)
        self._context = context
        return context

    def transform(self, env_obs: ObservationFrame) -> tuple[dict[str, np.ndarray], dict]:
        image64 = resize_chw_uint8(env_obs.image, IMAGE_SIZE)
        context = self._extract_context(env_obs)
        replay = {
            HYBRID_IMAGE_KEY: image64,
            HOUSE_CONTEXT_KEY: context.astype(np.float32),
        }
        agent_obs = {
            HYBRID_IMAGE_KEY: image64,
            HOUSE_CONTEXT_KEY: jnp.asarray(context, dtype=jnp.float32),
            "is_first": env_obs.is_first,
        }
        return replay, agent_obs
