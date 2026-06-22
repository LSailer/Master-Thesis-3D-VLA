"""Obs-batch contract tests for live VGGT context variants."""

import jax.numpy as jnp
import pytest

from src.r2dreamer.config import R2DreamerConfig
from src.r2dreamer.observation_preparation import build_vggt_contract
from src.r2dreamer.obs_batch import (
    CAMERA_POSE_KEY,
    GLOBAL_TOKENS_KEY,
    HYBRID_IMAGE_KEY,
    WORLD_POINTS_KEY,
    encoder_obs_from_batch,
)


def test_global_token_nogate_batch_keeps_tokens_singleton_and_flattens_images():
    """Global-token batches keep one live token set while flattening RGB frames."""
    cfg = R2DreamerConfig(
        encoder_type="vggt_house_global_tokens_nogate",
        obs_shape={HYBRID_IMAGE_KEY: (3, 64, 64), GLOBAL_TOKENS_KEY: (6, 8)},
        vggt_token_count=6,
        vggt_token_dim=8,
        compute_dtype="bfloat16",
    )
    batch = {
        "obs": {
            HYBRID_IMAGE_KEY: jnp.zeros((2, 3, 3, 64, 64), dtype=jnp.uint8),
            GLOBAL_TOKENS_KEY: jnp.ones((1, 6, 8), dtype=jnp.float32),
        }
    }

    encoder_obs = encoder_obs_from_batch(batch, cfg)

    assert set(encoder_obs) == {HYBRID_IMAGE_KEY, GLOBAL_TOKENS_KEY}
    assert encoder_obs[HYBRID_IMAGE_KEY].shape == (6, 3, 64, 64)
    assert encoder_obs[GLOBAL_TOKENS_KEY].shape == (1, 6, 8)
    assert encoder_obs[GLOBAL_TOKENS_KEY].dtype == jnp.bfloat16


def test_global_token_nogate_batch_rejects_broadcasted_replay_tokens():
    """Global-token replay rejects per-step token tensors."""
    cfg = R2DreamerConfig(
        encoder_type="vggt_house_global_tokens_nogate",
        obs_shape={HYBRID_IMAGE_KEY: (3, 64, 64), GLOBAL_TOKENS_KEY: (6, 8)},
        vggt_token_count=6,
        vggt_token_dim=8,
    )
    batch = {
        "obs": {
            HYBRID_IMAGE_KEY: jnp.zeros((2, 3, 3, 64, 64), dtype=jnp.uint8),
            GLOBAL_TOKENS_KEY: jnp.ones((2, 3, 6, 8), dtype=jnp.float32),
        }
    }

    with pytest.raises(ValueError, match="singleton"):
        encoder_obs_from_batch(batch, cfg)


def test_contract_layout_packs_flat_vggt_without_encoder_type_branch():
    """Contract layout drives flat WP/CP packing when encoder_type is unknown."""
    class FakeExtractor:
        """Small metadata-only VGGT extractor fake."""

        aggregator_feature_shape = (1374, 1024)
        image_size = 518
        wp_pool_size = 37

        def reset(self):
            """Match the VGGT extractor lifecycle surface."""

        def extract(self, image):
            """Match the VGGT extractor readout surface."""
            return image

    contract = build_vggt_contract(FakeExtractor(), feature_kind="wp_cp")
    snapshot = contract.to_snapshot()
    cfg = R2DreamerConfig(
        encoder_type="contract_only_alias",
        obs_shape=contract.encoder_input.buffer_shape(),
        encoder_input_contract=snapshot,
    )
    batch = {
        "obs": {
            WORLD_POINTS_KEY: jnp.ones((2, 3, 3, 37, 37), dtype=jnp.float16),
            CAMERA_POSE_KEY: jnp.ones((2, 3, 9), dtype=jnp.float16),
        }
    }

    encoder_obs = encoder_obs_from_batch(batch, cfg)

    assert encoder_obs.shape == (6, 4116)
    assert encoder_obs.dtype == jnp.bfloat16
