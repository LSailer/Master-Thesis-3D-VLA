"""VGGTHouseContextObsAdapter: RGB replay plus a live house-context vector."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from src.environments.observation import ObservationFrame
from src.r2dreamer.adapters.obs_adapter import ObsAdapter
from src.r2dreamer.adapters.token_adapters import FULL_TOKEN_SHAPE
from src.r2dreamer.encoders.constants import HOUSE_CONTEXT_DIM
from src.r2dreamer.encoders.transformer import TokenTransformerEncoder
from src.r2dreamer.observation_keys import (
    HOUSE_CONTEXT_KEY,
    HYBRID_IMAGE_KEY,
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
    VGGT_AGGREGATOR_TOKEN_COUNT,
    VGGT_FULL_TOKEN_EMBED_DIM,
)
from src.r2dreamer.observation_preparation.vggt_readouts import (
    full_aggregator_tokens,
)
from src.shared.video_utils import resize_chw_uint8

HOUSE_CONTEXT_FEATURE_DIM = HYBRID_RGB_DIM + HOUSE_CONTEXT_DIM


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
