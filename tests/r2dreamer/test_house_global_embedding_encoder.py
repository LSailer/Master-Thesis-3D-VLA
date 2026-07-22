"""CPU tests for the PointNet-reducer house global embedding encoder.

Vanilla PointNet reducer over VGGT global patch tokens. The first branch is
selected by what the obs dict carries: an RGB frame goes through the conv
encoder, otherwise the camera token rides its own side branch
(src/prototyp/house_global_embedding/IDEA.md).
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from src.r2dreamer.encoders.mlp import (
    HouseGlobalEmbeddingEncoder,
    HouseGlobalObs,
    TokenReducer,
)
from src.r2dreamer.observation_keys import (
    CAMERA_TOKEN_GLOBAL_KEY,
    GLOBAL_PATCH_TOKENS_KEY,
    HYBRID_IMAGE_KEY,
)

# Both branches are fixed at TokenReducer's default output_dim; the conv
# encoder emits the same width for a (64, 64, 3) frame. Only the hidden width
# and layer count are configurable on the encoder.
BRANCH_DIM = 1024
TOKEN_DIM = 16
N_PATCHES = 64


def _make_encoder(**overrides):
    kwargs = {"mlp_layers": 1, "hidden_dim": 16}
    kwargs.update(overrides)
    return HouseGlobalEmbeddingEncoder(**kwargs)


def _obs(batch, *, image=False, camera=False, n_patches=N_PATCHES, key=0):
    """Build an obs dict carrying the requested optional branch fields."""
    k_patch, k_cam, k_img = jax.random.split(jax.random.PRNGKey(key), 3)
    obs = {
        GLOBAL_PATCH_TOKENS_KEY: jax.random.normal(
            k_patch, (batch, n_patches, TOKEN_DIM), dtype=jnp.float32
        )
    }
    if camera:
        obs[CAMERA_TOKEN_GLOBAL_KEY] = jax.random.normal(
            k_cam, (batch, 1, TOKEN_DIM), dtype=jnp.float32
        )
    if image:
        obs[HYBRID_IMAGE_KEY] = jax.random.uniform(
            k_img, (batch, 64, 64, 3), dtype=jnp.float32
        )
    return obs


# --- Edge case 1: the obs dict carries an image ---------------------------


def test_image_obs_broadcasts_singleton_patch_tokens():
    """Live augment injects (N, D) patches alongside (B, T, 3, H, W) RGB."""
    enc = _make_encoder()
    k_img, k_patch = jax.random.split(jax.random.PRNGKey(9))
    obs = {
        HYBRID_IMAGE_KEY: jax.random.uniform(
            k_img, (2, 4, 64, 64, 3), dtype=jnp.float32
        ),
        GLOBAL_PATCH_TOKENS_KEY: jax.random.normal(
            k_patch, (N_PATCHES, TOKEN_DIM), dtype=jnp.float32
        ),
    }
    params = enc.init(jax.random.PRNGKey(1), obs)
    out = enc.apply(params, obs)
    assert out.shape == (2, 4, 2 * BRANCH_DIM)
    assert bool(jnp.isfinite(out).all())


def test_image_obs_uses_the_rgb_branch():
    enc = _make_encoder()
    obs = _obs(5, image=True)
    params = enc.init(jax.random.PRNGKey(1), obs)
    out = enc.apply(params, obs)
    assert out.shape == (5, 2 * BRANCH_DIM)
    assert bool(jnp.isfinite(out).all())
    assert set(params["params"]) == {"house", "rgb"}


def test_image_shadows_the_camera_token_when_both_are_present():
    """The image check runs first. This is the production shape: the L1 adapter
    stores the RGB frame (the decoder's target) alongside both token fields, so
    a dict carrying all three still takes the RGB branch."""
    enc = _make_encoder()
    obs = _obs(2, image=True, camera=True)
    params = enc.init(jax.random.PRNGKey(1), obs)
    out = enc.apply(params, obs)
    assert out.shape == (2, 2 * BRANCH_DIM)
    assert set(params["params"]) == {"house", "rgb"}


# --- Edge case 2: no image, camera token present --------------------------


def test_camera_obs_without_image_uses_the_camera_branch():
    enc = _make_encoder()
    obs = _obs(5, camera=True)
    params = enc.init(jax.random.PRNGKey(1), obs)
    out = enc.apply(params, obs)
    assert out.shape == (5, 2 * BRANCH_DIM)
    assert bool(jnp.isfinite(out).all())
    assert set(params["params"]) == {"house", "camera"}


def test_camera_token_is_isolated_from_house_branch():
    """The camera token rides its own side branch: changing it must move only
    the camera half of the fused embedding, and changing a patch token must
    move only the house half (the deliberate split from IDEA.md)."""
    enc = _make_encoder()
    obs = _obs(1, camera=True)
    params = enc.init(jax.random.PRNGKey(1), obs)
    before = enc.apply(params, obs)
    cam_before, house_before = before[..., :BRANCH_DIM], before[..., BRANCH_DIM:]

    # Change only the camera token.
    obs_cam = {**obs, CAMERA_TOKEN_GLOBAL_KEY: obs[CAMERA_TOKEN_GLOBAL_KEY] + 1.0}
    after_cam = enc.apply(params, obs_cam)
    assert bool(jnp.any(jnp.abs(after_cam[..., :BRANCH_DIM] - cam_before) > 1e-4))
    np.testing.assert_allclose(after_cam[..., BRANCH_DIM:], house_before, atol=1e-5)

    # Change only one patch token.
    obs_patch = {
        **obs,
        GLOBAL_PATCH_TOKENS_KEY: obs[GLOBAL_PATCH_TOKENS_KEY].at[:, 0, :].add(1.0),
    }
    after_patch = enc.apply(params, obs_patch)
    np.testing.assert_allclose(after_patch[..., :BRANCH_DIM], cam_before, atol=1e-5)
    assert bool(
        jnp.any(jnp.abs(after_patch[..., BRANCH_DIM:] - house_before) > 1e-4)
    )


# --- Edge case 3: neither optional field ----------------------------------


def test_patches_only_obs_returns_the_house_embedding():
    enc = _make_encoder()
    obs = _obs(5)
    params = enc.init(jax.random.PRNGKey(1), obs)
    out = enc.apply(params, obs)
    assert out.shape == (5, BRANCH_DIM)
    assert bool(jnp.isfinite(out).all())
    assert set(params["params"]) == {"house"}


def test_missing_patch_tokens_is_a_keyerror():
    """global_patch_tokens is the one mandatory field, so it fails loud at the
    dict boundary instead of dying inside the reducer."""
    enc = _make_encoder()
    obs = _obs(2, camera=True)
    del obs[GLOBAL_PATCH_TOKENS_KEY]
    with pytest.raises(KeyError, match=GLOBAL_PATCH_TOKENS_KEY):
        enc.init(jax.random.PRNGKey(1), obs)


# --- Shared contract ------------------------------------------------------


def test_house_global_obs_passes_through_without_mapping():
    """A prebuilt HouseGlobalObs must bypass the dict coercion untouched."""
    enc = _make_encoder()
    dict_obs = _obs(2, camera=True)
    struct_obs = HouseGlobalObs(
        global_patch_tokens=dict_obs[GLOBAL_PATCH_TOKENS_KEY],
        camera_token_global=dict_obs[CAMERA_TOKEN_GLOBAL_KEY],
    )
    params = enc.init(jax.random.PRNGKey(1), struct_obs)
    np.testing.assert_allclose(
        enc.apply(params, struct_obs), enc.apply(params, dict_obs), atol=1e-6
    )


def test_preserves_replay_leading_dims():
    enc = _make_encoder()
    # Mimic a sampled replay batch (B, T, ...): build the (2, 4, ...) tensors
    # directly so they are concrete (broadcast_to views cannot be reshaped).
    k_cam, k_patch = jax.random.split(jax.random.PRNGKey(2))
    obs_bt = {
        CAMERA_TOKEN_GLOBAL_KEY: jax.random.normal(
            k_cam, (2, 4, 1, TOKEN_DIM), dtype=jnp.float32
        ),
        GLOBAL_PATCH_TOKENS_KEY: jax.random.normal(
            k_patch, (2, 4, N_PATCHES, TOKEN_DIM), dtype=jnp.float32
        ),
    }
    params = enc.init(jax.random.PRNGKey(1), obs_bt)
    out = enc.apply(params, obs_bt)
    assert out.shape == (2, 4, 2 * BRANCH_DIM)


def test_jit_and_grad():
    enc = _make_encoder()
    obs = _obs(2, camera=True)
    params = enc.init(jax.random.PRNGKey(1), obs)

    @jax.jit
    def loss_fn(p):
        return jnp.sum(enc.apply(p, obs) ** 2)

    grads = jax.grad(loss_fn)(params)
    leaves = jax.tree_util.tree_leaves(grads)
    assert leaves
    assert all(bool(jnp.isfinite(g).all()) for g in leaves)
    # Both the reducer and the camera side branch must receive gradient.
    assert bool((jnp.abs(grads["params"]["house"]["proj"]["kernel"]) > 0).any())
    assert bool((jnp.abs(grads["params"]["camera"]["proj"]["kernel"]) > 0).any())
    # The per-token reducer MLP must receive gradient signal.
    assert bool((jnp.abs(grads["params"]["house"]["hidden0"]["kernel"]) > 0).any())


def test_max_pool_is_permutation_invariant():
    """PointNet's symmetric g-pool: permuting the patch tokens must not change
    the house embedding."""
    enc = _make_encoder()
    obs = _obs(1)
    params = enc.init(jax.random.PRNGKey(1), obs)
    house_before = enc.apply(params, obs)

    perm = jax.random.permutation(jax.random.PRNGKey(7), N_PATCHES)
    obs_perm = {**obs, GLOBAL_PATCH_TOKENS_KEY: obs[GLOBAL_PATCH_TOKENS_KEY][:, perm, :]}
    house_after = enc.apply(params, obs_perm)
    np.testing.assert_allclose(house_before, house_after, atol=1e-5)


def test_reducer_nonlinearity_before_pool_is_present():
    """Guards the design caveat: a Dense->silu before the max-pool is required,
    otherwise two linear layers around a linear pool collapse to one Linear."""
    enc = _make_encoder()
    params = enc.init(jax.random.PRNGKey(1), _obs(1, camera=True))
    # The reducer MLP and its RMSNorm must exist as trained parameters.
    assert "hidden0" in params["params"]["house"]
    assert "norm0" in params["params"]["house"]


def test_mlp_layers_controls_reducer_depth():
    enc = _make_encoder(mlp_layers=3)
    params = enc.init(jax.random.PRNGKey(1), _obs(1, camera=True))
    for i in range(3):
        assert f"hidden{i}" in params["params"]["house"]
        assert f"norm{i}" in params["params"]["house"]
    assert "hidden3" not in params["params"]["house"]


def test_token_reducer_pools_over_tokens_but_squeezes_singletons():
    """pool_tokens=True max-pools the token axis; False squeezes a singleton."""
    tokens = _obs(2)[GLOBAL_PATCH_TOKENS_KEY]
    pooled = TokenReducer(output_dim=8, hidden=16, layers=1)
    p = pooled.init(jax.random.PRNGKey(0), tokens)
    assert pooled.apply(p, tokens).shape == (2, 8)

    single = _obs(2, camera=True)[CAMERA_TOKEN_GLOBAL_KEY]
    squeezed = TokenReducer(output_dim=8, hidden=16, layers=1, pool_tokens=False)
    p = squeezed.init(jax.random.PRNGKey(0), single)
    assert squeezed.apply(p, single).shape == (2, 8)


def test_short_training_loop_is_stable():
    enc = _make_encoder()
    obs = _obs(4, camera=True)
    target = jax.random.normal(
        jax.random.PRNGKey(3), (4, 2 * BRANCH_DIM), dtype=jnp.float32
    )
    params = enc.init(jax.random.PRNGKey(1), obs)
    opt = optax.adam(1e-3)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state):
        def loss_fn(p):
            return jnp.mean((enc.apply(p, obs) - target) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        gnorm = optax.global_norm(grads)
        updates, opt_state = opt.update(grads, opt_state)
        return optax.apply_updates(params, updates), opt_state, loss, gnorm

    losses = []
    for _ in range(15):
        params, opt_state, loss, gnorm = step(params, opt_state)
        assert bool(jnp.isfinite(loss)), "loss went non-finite"
        assert bool(jnp.isfinite(gnorm)), "grad norm went non-finite"
        losses.append(float(loss))
    assert losses[-1] < losses[0]
