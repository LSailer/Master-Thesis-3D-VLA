"""Obs-batch contract tests for live VGGT context variants."""

import jax.numpy as jnp
import pytest

from src.r2dreamer.config import R2DreamerConfig
from src.r2dreamer.obs_batch import (
    GLOBAL_TOKENS_KEY,
    HYBRID_IMAGE_KEY,
    encoder_obs_from_batch,
)


def test_global_token_nogate_batch_keeps_tokens_singleton_and_flattens_images():
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
