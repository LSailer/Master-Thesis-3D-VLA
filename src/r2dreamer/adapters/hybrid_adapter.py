"""HybridObsAdapter: wraps a VGGT extractor for the CNN+WP/CP hybrid encoder."""

from __future__ import annotations

import jax.numpy as jnp
import jax
import numpy as np

from src.r2dreamer.adapters.obs_adapter import ObsAdapter
from src.r2dreamer.adapters.vggt_adapter import (
    AGG_TOKEN_TOKENS,
    VGGT_FEATURE_DIM,
    full_aggregator_tokens,
    flatten_world_points_camera_pose,
)
from src.r2dreamer.obs_batch import (
    HOUSE_CONTEXT_KEY,
    HYBRID_IMAGE_KEY,
    HYBRID_WP_CP_KEY,
)
from src.shared.video_utils import resize_chw_uint8
from src.r2dreamer.world_model.encoders import (
    HOUSE_CONTEXT_DIM,
    VGGTFullTokenContextTransformer,
)


# Derived, not hand-typed: RGB branch (3*64*64, flattened) + the VGGT WP/CP vector.
# VGGT_FEATURE_DIM is the single source of truth for the 4116 term (see vggt_adapter),
# so a grid-size ablation that changes it stays consistent here automatically.
HYBRID_FEATURE_DIM = 3 * 64 * 64 + VGGT_FEATURE_DIM  # 12288 RGB + 4116 WP/CP = 16404
HYBRID_IMAGE_SHAPE = (3, 64, 64)
HOUSE_CONTEXT_FEATURE_DIM = 3 * 64 * 64 + HOUSE_CONTEXT_DIM
FULL_TOKEN_SHAPE = (AGG_TOKEN_TOKENS, 2048)


class HybridObsAdapter(ObsAdapter):
    """Builds the hybrid replay fields ``{"image": rgb64, "wp_cp": wp_cp}``.

    The env renders 518x518 CHW uint8 (for VGGT). Each step we run VGGT once to
    obtain world_points + camera_pose, downsample the same frame to 64x64 for the
    CNN branch, and store both modalities under explicit replay keys. The agent
    still packs them into the legacy flat encoder input at the JAX boundary.
    """

    def __init__(self, extractor):
        super().__init__(
            buffer_dtype={HYBRID_IMAGE_KEY: "uint8", HYBRID_WP_CP_KEY: "float32"},
            buffer_shape={
                HYBRID_IMAGE_KEY: HYBRID_IMAGE_SHAPE,
                HYBRID_WP_CP_KEY: (VGGT_FEATURE_DIM,),
            },
            normalize_on_sample={HYBRID_IMAGE_KEY: False, HYBRID_WP_CP_KEY: False},
            agent_obs_shape=(HYBRID_FEATURE_DIM,),
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
        self._context_transformer = context_transformer or VGGTFullTokenContextTransformer()
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
        if tuple(tokens.shape) != FULL_TOKEN_SHAPE:
            raise ValueError(
                f"expected VGGT full-token shape {FULL_TOKEN_SHAPE}, got {tokens.shape}"
            )
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
