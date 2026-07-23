"""Parametrized CPU test over the encoder RECIPES registry (migration step 2).

For every registered recipe: build a *fake* prepared frame (never load VGGT),
infer the obs spec from it, run the fail-fast branch-key check, init the
composite, and assert ``apply`` returns ``f32[B, T, E]`` on a replay-shaped
batch. This is the generic shape/plumbing contract; per-encoder numeric parity
lives in the variant-specific tests until they are consolidated here at the end
(DELETIONS.md).
"""

import jax
import jax.numpy as jnp
import pytest

from src.configs.config import R2DreamerConfig
from src.r2dreamer.encoders.composite import CompositeEncoder
from src.r2dreamer.encoders.recipes import (
    RECIPES,
    check_branch_keys,
    infer_obs_spec,
)
from src.r2dreamer.observation_keys import HYBRID_IMAGE_KEY, HYBRID_WP_CP_KEY

HYBRID_VGGT_DIM = 4116

# Fake prepared frames (batch prefix 1), one per recipe. Hand-built so CPU tests
# never touch the VGGT extractor (PROBLEMS.md).
FAKE_FRAMES = {
    "cnn": {HYBRID_IMAGE_KEY: jnp.zeros((1, 64, 64, 3), jnp.float32)},
    "hybrid": {
        HYBRID_IMAGE_KEY: jnp.zeros((1, 64, 64, 3), jnp.float32),
        HYBRID_WP_CP_KEY: jnp.zeros((1, HYBRID_VGGT_DIM), jnp.float32),
    },
}


def _config_for(encoder_type: str) -> R2DreamerConfig:
    return R2DreamerConfig(encoder_type=encoder_type)


def test_registry_covers_cnn_and_hybrid():
    assert {"cnn", "hybrid"} <= set(RECIPES)


@pytest.mark.parametrize("name", sorted(RECIPES))
def test_recipe_fake_frame_infer_init_apply(name):
    recipe = RECIPES[name]
    assert name in FAKE_FRAMES, f"add a fake frame for recipe {name!r}"
    frame = FAKE_FRAMES[name]
    cfg = _config_for(recipe.encoder_type)

    composite = recipe.build_composite(cfg)

    # Infer obs spec from the first prepared frame; then the one startup check.
    obs_spec = infer_obs_spec(frame)
    check_branch_keys(composite, obs_spec.keys())

    enc = CompositeEncoder(composite)
    params = enc.init(jax.random.PRNGKey(0), frame)

    # Replay-shaped batch: (B, T, *event) per key.
    B, T = 2, 3
    batch = {k: jnp.zeros((B, T, *shape), jnp.float32) for k, shape in obs_spec.items()}
    out = enc.apply(params, batch)

    assert out.ndim == 3
    assert out.shape[:2] == (B, T)
    assert out.shape[2] > 0
    assert out.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(out)))


def test_check_branch_keys_rejects_mismatch():
    recipe = RECIPES["hybrid"]
    composite = recipe.build_composite(_config_for("hybrid"))
    with pytest.raises(ValueError, match="key mismatch"):
        check_branch_keys(composite, {HYBRID_IMAGE_KEY})  # missing wp_cp


def test_rgb_key_marks_decoder_capable_recipes():
    assert RECIPES["cnn"].rgb_key == HYBRID_IMAGE_KEY
    assert RECIPES["hybrid"].rgb_key == HYBRID_IMAGE_KEY
