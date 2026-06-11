"""HybridObsAdapter: wraps a VGGT extractor for the CNN+WP/CP hybrid encoder."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from src.r2dreamer.adapters.obs_adapter import ObsAdapter
from src.r2dreamer.obs_batch import HOUSE_CONTEXT_KEY, HYBRID_IMAGE_KEY
from src.r2dreamer.adapters.vggt_adapter import (
    VGGT_FEATURE_DIM,
    flatten_world_points_camera_pose,
)
from src.shared.video_utils import resize_chw_uint8


# Derived, not hand-typed: RGB branch (3*64*64, flattened) + the VGGT WP/CP vector.
# VGGT_FEATURE_DIM is the single source of truth for the 4116 term (see vggt_adapter),
# so a grid-size ablation that changes it stays consistent here automatically.
HYBRID_FEATURE_DIM = 3 * 64 * 64 + VGGT_FEATURE_DIM  # 12288 RGB + 4116 WP/CP = 16404


class HybridObsAdapter(ObsAdapter):
    """Builds the flat hybrid buffer vector ``[ rgb (12288) | wp_cp (4116) ]``.

    The env renders 518x518 CHW uint8 (for VGGT). Each step we run VGGT once to
    obtain world_points + camera_pose, downsample the same frame to 64x64 for the
    CNN branch, and concatenate the normalised RGB ([0,1]) with the wp/cp vector.
    RGB is already divided by 255 here, so ``normalize_on_sample=False``.
    """

    def __init__(self, extractor):
        super().__init__(
            buffer_dtype="float32",
            buffer_shape=(HYBRID_FEATURE_DIM,),
            normalize_on_sample=False,
            on_episode_reset=extractor.reset,
        )
        self._extractor = extractor

    def transform(self, obs_dict: dict) -> tuple[np.ndarray, dict]:
        out = self._extractor.extract(obs_dict["image"])  # image is 518 CHW uint8
        wp_cp = flatten_world_points_camera_pose(out)  # jnp (4116,)
        img64 = resize_chw_uint8(obs_dict["image"], 64)  # (3,64,64) uint8
        rgb = (img64.astype(np.float32) / 255.0).reshape(-1)  # (12288,) [0,1]
        replay = np.concatenate(
            [rgb, np.asarray(wp_cp, dtype=np.float32)]
        ).astype(np.float32)  # (16404,)
        agent_obs = {
            "hybrid": jnp.asarray(replay),
            "image": img64,
            "is_first": obs_dict.get("is_first", False),
        }
        return replay, agent_obs


class VGGTHouseContextObsAdapter(ObsAdapter):
    """RGB replay plus live InfiniteVGGT house context for L1 experiments.

    Replay stores only the 64x64 RGB frame. The VGGT extractor remains live
    across episode resets; its bounded streaming cache supplies a current
    house-level WP/CP readout that is injected into sampled replay windows.
    """

    def __init__(self, extractor):
        super().__init__(
            buffer_dtype="uint8",
            buffer_shape=(3, 64, 64),
            normalize_on_sample=True,
            on_episode_reset=None,
        )
        self._extractor = extractor
        self._context: np.ndarray | None = None
        self.agent_obs_shape = (HYBRID_FEATURE_DIM,)

    def _extract_context(self, image: np.ndarray) -> np.ndarray:
        out = self._extractor.extract(image)
        context = np.asarray(flatten_world_points_camera_pose(out), dtype=np.float32)
        if context.shape != (VGGT_FEATURE_DIM,):
            raise ValueError(
                f"expected VGGT house context shape {(VGGT_FEATURE_DIM,)}, "
                f"got {context.shape}"
            )
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
        context = jnp.broadcast_to(context, (*image.shape[:2], VGGT_FEATURE_DIM))
        return {
            **batch,
            "obs": {
                HYBRID_IMAGE_KEY: image,
                HOUSE_CONTEXT_KEY: context,
            },
        }
