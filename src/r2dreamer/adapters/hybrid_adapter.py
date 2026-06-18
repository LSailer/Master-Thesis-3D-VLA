"""HybridObsAdapter: wraps a VGGT extractor for the CNN+WP/CP hybrid encoder."""

from __future__ import annotations

import jax.numpy as jnp
import jax
import numpy as np

from src.r2dreamer.adapters.obs_adapter import ObsAdapter
from src.r2dreamer.observation_preparation.vggt_readouts import (
    flatten_world_points_camera_pose,
    full_aggregator_tokens,
)
from src.r2dreamer.obs_batch import (
    FULL_TOKENS_KEY,
    HOUSE_CONTEXT_KEY,
    HYBRID_IMAGE_KEY,
    HYBRID_WP_CP_KEY,
)
from src.r2dreamer.observation_preparation.vggt import (
    HYBRID_IMAGE_SHAPE,
    HYBRID_RGB_DIM,
    VGGT_AGGREGATOR_TOKEN_COUNT,
    VGGT_FULL_TOKEN_EMBED_DIM,
    build_hybrid_contract,
)
from src.shared.video_utils import resize_chw_uint8
from src.r2dreamer.world_model.encoders import (
    HOUSE_CONTEXT_DIM,
    VGGTFullTokenContextTransformer,
)


HOUSE_CONTEXT_FEATURE_DIM = HYBRID_RGB_DIM + HOUSE_CONTEXT_DIM
FULL_TOKEN_SHAPE = (VGGT_AGGREGATOR_TOKEN_COUNT, VGGT_FULL_TOKEN_EMBED_DIM)


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

    def transform(self, obs_dict: dict) -> tuple[dict[str, np.ndarray], dict]:
        out = self._extractor.extract(obs_dict["image"])  # image is 518 CHW uint8
        wp_cp = flatten_world_points_camera_pose(out)  # jnp (4116,)
        img64 = resize_chw_uint8(obs_dict["image"], 64)  # (3,64,64) uint8
        replay = {
            HYBRID_IMAGE_KEY: img64,
            HYBRID_WP_CP_KEY: np.asarray(wp_cp, dtype=np.float32),
        }
        agent_obs = {
            HYBRID_IMAGE_KEY: img64,
            HYBRID_WP_CP_KEY: jnp.asarray(replay[HYBRID_WP_CP_KEY]),
            "is_first": obs_dict.get("is_first", False),
        }
        return replay, agent_obs


class VGGTHouseFullTokenObsAdapter(ObsAdapter):
    """RGB replay plus live full VGGT tokens for the no-gate Transformer test.

    Replay stores only the 64x64 RGB frame. The latest live ``(1374, 2048)``
    InfiniteVGGT full-token tensor is cached outside replay and injected into
    sampled windows, so the trainable agent encoder can learn from live tokens
    without storing them in the buffer.
    """

    def __init__(self, extractor):
        super().__init__(
            buffer_dtype="uint8",
            buffer_shape=(3, 64, 64),
            normalize_on_sample=True,
            agent_obs_shape={
                HYBRID_IMAGE_KEY: HYBRID_IMAGE_SHAPE,
                FULL_TOKENS_KEY: FULL_TOKEN_SHAPE,
            },
            on_episode_reset=None,
        )
        self._extractor = extractor
        self._tokens: np.ndarray | None = None

    def _extract_tokens(self, image: np.ndarray) -> np.ndarray:
        out = self._extractor.extract(image)
        tokens = full_aggregator_tokens(out, FULL_TOKEN_SHAPE)
        self._tokens = np.asarray(tokens, dtype=np.float32)
        return self._tokens

    def transform(self, obs_dict: dict) -> tuple[np.ndarray, dict]:
        image64 = resize_chw_uint8(obs_dict["image"], 64)
        tokens = self._extract_tokens(obs_dict["image"])
        agent_obs = {
            HYBRID_IMAGE_KEY: image64,
            FULL_TOKENS_KEY: jnp.asarray(tokens, dtype=jnp.float32),
            "is_first": obs_dict.get("is_first", False),
        }
        return image64, agent_obs

    def augment_replay_batch(self, batch: dict) -> dict:
        if self._tokens is None:
            raise RuntimeError(
                "VGGTHouseFullTokenObsAdapter has no live full tokens yet; "
                "call transform() before sampling replay."
            )
        image = batch["obs"]
        tokens = jnp.asarray(self._tokens, dtype=jnp.float32)
        tokens = jnp.broadcast_to(tokens, (*image.shape[:2], *FULL_TOKEN_SHAPE))
        return {
            **batch,
            "obs": {
                HYBRID_IMAGE_KEY: image,
                FULL_TOKENS_KEY: tokens,
            },
        }


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
        context_transformer: VGGTFullTokenContextTransformer | None = None,
        rng_seed: int = 0,
    ):
        super().__init__(
            buffer_dtype="uint8",
            buffer_shape=(3, 64, 64),
            normalize_on_sample=True,
            on_episode_reset=None,
        )
        self._extractor = extractor
        self._context_transformer = (
            context_transformer or VGGTFullTokenContextTransformer()
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

    def _extract_context(self, image: np.ndarray) -> np.ndarray:
        out = self._extractor.extract(image)
        tokens = full_aggregator_tokens(out, FULL_TOKEN_SHAPE)
        context = self._project_context(tokens)
        self._context = context
        return context

    def transform(self, obs_dict: dict) -> tuple[np.ndarray, dict]:
        image64 = resize_chw_uint8(obs_dict["image"], 64)
        context = self._extract_context(obs_dict["image"])
        agent_obs = {
            HYBRID_IMAGE_KEY: image64,
            HOUSE_CONTEXT_KEY: jnp.asarray(context, dtype=jnp.float32),
            "is_first": obs_dict.get("is_first", False),
        }
        return image64, agent_obs

    def augment_replay_batch(self, batch: dict) -> dict:
        if self._context is None:
            raise RuntimeError(
                "VGGTHouseContextObsAdapter has no live house context yet; "
                "call transform() before sampling replay."
            )
        image = batch["obs"]
        context = jnp.asarray(self._context, dtype=jnp.float32)
        context = jnp.broadcast_to(context, (*image.shape[:2], HOUSE_CONTEXT_DIM))
        return {
            **batch,
            "obs": {
                HYBRID_IMAGE_KEY: image,
                HOUSE_CONTEXT_KEY: context,
            },
        }
