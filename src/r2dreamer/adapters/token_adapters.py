"""RGB-replay adapters that store live VGGT aggregator tokens.

Three closely related L1 experiments that all replay the 64x64 RGB frame and,
per step, store one slice of the VGGT aggregator's token stream for the agent:

* :class:`VGGTHouseFullTokenObsAdapter` — the full ``(1374, 2048)`` token map.
* :class:`VGGTHouseGlobalTokenObsAdapter` — the singleton global-half tokens.
* :class:`VGGTHouseGlobalEmbeddingObsAdapter` — the global-half tokens split
  into the camera token and the patch tokens for the PointNet reducer.

The first two share the :class:`_RGBLiveTokenObsAdapter` base (single token
field, fixed shape check); the third stores two split fields and drives
optional diagnostic PLY dumps.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

import jax.numpy as jnp
import numpy as np

from src.environments.observation import ObservationFrame
from src.r2dreamer.adapters.obs_adapter import ObsAdapter
from src.r2dreamer.observation_keys import (
    CAMERA_TOKEN_GLOBAL_KEY,
    FULL_TOKENS_KEY,
    GLOBAL_PATCH_TOKENS_KEY,
    GLOBAL_TOKENS_KEY,
    HYBRID_IMAGE_KEY,
)
from src.r2dreamer.observation_preparation.vggt import (
    HYBRID_IMAGE_SHAPE as IMAGE_SHAPE,
)
from src.r2dreamer.observation_preparation.vggt import (
    HYBRID_IMAGE_SIZE as IMAGE_SIZE,
)
from src.r2dreamer.observation_preparation.vggt import (
    VGGT_AGGREGATOR_EMBED_DIM,
    VGGT_AGGREGATOR_PATCH_START_IDX,
    VGGT_AGGREGATOR_TOKEN_COUNT,
    VGGT_FULL_TOKEN_EMBED_DIM,
)
from src.r2dreamer.observation_preparation.vggt_readouts import (
    VGGTOutputLike,
    full_aggregator_tokens,
)
from src.shared.video_utils import resize_chw_uint8

FULL_TOKEN_SHAPE = (VGGT_AGGREGATOR_TOKEN_COUNT, VGGT_FULL_TOKEN_EMBED_DIM)
GLOBAL_TOKEN_SHAPE = (VGGT_AGGREGATOR_TOKEN_COUNT, VGGT_AGGREGATOR_EMBED_DIM)
# House-global-embedding L1 split of the global-half tokens (1374, 1024):
# camera token [0:1] and patch tokens [5:] (4 registers dropped).
CAMERA_TOKEN_GLOBAL_SHAPE = (1, VGGT_AGGREGATOR_EMBED_DIM)
GLOBAL_PATCH_TOKENS_SHAPE = (
    VGGT_AGGREGATOR_TOKEN_COUNT - VGGT_AGGREGATOR_PATCH_START_IDX,
    VGGT_AGGREGATOR_EMBED_DIM,
)


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
        self._dump_every = int(pointcloud_dump_every)
        self._dump_dir = pointcloud_dump_dir
        self._dump_enabled = (
            self._dump_every > 0
            and self._dump_dir is not None
            and hasattr(extractor, "write_point_cloud_ply")
        )
        self._env_steps = 0
        self._episode_starts_seen = 0
        self._first_episode_dumped = False
        self._dump_count = 0

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

    def _maybe_dump_pointcloud(self, label: str) -> None:
        """Write a PLY snapshot via the extractor if dumping is enabled.

        Args:
          label: Filename label inserted into ``pointcloud_<label>.ply``.
        """
        if not self._dump_enabled:
            return
        os.makedirs(self._dump_dir, exist_ok=True)
        path = os.path.join(self._dump_dir, f"pointcloud_{label}.ply")
        self._extractor.write_point_cloud_ply(path)
        self._dump_count += 1

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
        if env_obs.is_first:
            self._episode_starts_seen += 1
            if (
                self._episode_starts_seen == 2
                and not self._first_episode_dumped
            ):
                self._maybe_dump_pointcloud("end_of_first_episode")
                self._first_episode_dumped = True

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

        self._env_steps += 1
        if self._dump_enabled and self._env_steps % self._dump_every == 0:
            self._maybe_dump_pointcloud(f"step{self._env_steps}")
        return replay, agent_obs

    def diagnostics(self) -> dict[str, float]:
        """Return PLY-dump and extractor-cache diagnostics for the run summary.

        Returns:
          A dict with ``dump_count``, ``env_steps``, and
          ``camera_head_cache_active`` (0 = unallocated, as required under
          PERSIST_SCENE with heads off).
        """
        stats = {
            "house_global_embedding/dump_count": float(self._dump_count),
            "house_global_embedding/env_steps": float(self._env_steps),
        }
        # Camera-head cache must stay unallocated under PERSIST_SCENE with
        # heads off (risk #2 in PROBLEMS.md): expose its state for the smoke.
        cache = getattr(self._extractor, "_past_kvs_camera", "missing")
        stats["house_global_embedding/camera_head_cache_active"] = (
            0.0 if cache is None else 1.0 if cache != "missing" else -1.0
        )
        return stats
