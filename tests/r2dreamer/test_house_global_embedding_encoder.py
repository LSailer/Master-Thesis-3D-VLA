"""CPU tests for the PointNet-reducer house global embedding encoder.

Covers src/r2dreamer/encoders/mlp.py::HouseGlobalEmbeddingEncoder — the run-1
vanilla PointNet reducer over VGGT global patch tokens with the camera token on
its own side branch (src/prototyp/house_global_embedding/IDEA.md).
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from src.r2dreamer.encoders.mlp import HouseGlobalEmbeddingEncoder
from src.r2dreamer.observation_keys import (
    CAMERA_TOKEN_GLOBAL_KEY,
    GLOBAL_PATCH_TOKENS_KEY,
)


def _make_encoder(**overrides):
    kwargs = dict(
        embed_dim=32,
        token_dim=16,
        num_patch_tokens=64,
        reducer_hidden=16,
        reducer_layers=1,
        camera_hidden=16,
        camera_layers=1,
    )
    kwargs.update(overrides)
    return HouseGlobalEmbeddingEncoder(**kwargs)


def _make_obs(batch: int, n_patches: int, token_dim: int, *, key=0):
    rng = jax.random.PRNGKey(key)
    k_cam, k_patch = jax.random.split(rng)
    return {
        CAMERA_TOKEN_GLOBAL_KEY: jax.random.normal(
            k_cam, (batch, 1, token_dim), dtype=jnp.float32
        ),
        GLOBAL_PATCH_TOKENS_KEY: jax.random.normal(
            k_patch, (batch, n_patches, token_dim), dtype=jnp.float32
        ),
    }


def test_forward_shape_and_finite():
    enc = _make_encoder()
    obs = _make_obs(batch=5, n_patches=64, token_dim=16)
    params = enc.init(jax.random.PRNGKey(1), obs)
    out = enc.apply(params, obs)
    assert out.shape == (5, 64)  # 2 * embed_dim
    assert bool(jnp.isfinite(out).all())


def test_branches_split_is_camera_then_house():
    enc = _make_encoder()
    obs = _make_obs(batch=3, n_patches=64, token_dim=16)
    params = enc.init(jax.random.PRNGKey(1), obs)
    camera_embed, house_embed = enc.apply(params, obs, method="branches")
    fused = enc.apply(params, obs)
    assert camera_embed.shape == (3, 32)
    assert house_embed.shape == (3, 32)
    np.testing.assert_allclose(
        fused, jnp.concatenate([camera_embed, house_embed], axis=-1)
    )


def test_preserves_replay_leading_dims():
    enc = _make_encoder()
    # Mimic a sampled replay batch (B, T, ...): build the (2, 4, ...) tensors
    # directly so they are concrete (broadcast_to views cannot be reshaped).
    rng = jax.random.PRNGKey(2)
    k_cam, k_patch = jax.random.split(rng)
    obs_bt = {
        CAMERA_TOKEN_GLOBAL_KEY: jax.random.normal(
            k_cam, (2, 4, 1, 16), dtype=jnp.float32
        ),
        GLOBAL_PATCH_TOKENS_KEY: jax.random.normal(
            k_patch, (2, 4, 64, 16), dtype=jnp.float32
        ),
    }
    params = enc.init(jax.random.PRNGKey(1), obs_bt)
    out = enc.apply(params, obs_bt)
    assert out.shape == (2, 4, 64)


def test_jit_and_grad():
    enc = _make_encoder()
    obs = _make_obs(batch=2, n_patches=64, token_dim=16)
    params = enc.init(jax.random.PRNGKey(1), obs)

    @jax.jit
    def loss_fn(p):
        return jnp.sum(enc.apply(p, obs) ** 2)

    grads = jax.grad(loss_fn)(params)
    leaves = jax.tree_util.tree_leaves(grads)
    assert leaves
    assert all(bool(jnp.isfinite(g).all()) for g in leaves)
    # Both the reducer and the camera side branch must receive gradient.
    assert bool((jnp.abs(grads["params"]["house_proj"]["kernel"]) > 0).any())
    assert bool((jnp.abs(grads["params"]["camera_proj"]["kernel"]) > 0).any())
    # The per-token reducer MLP must receive gradient signal.
    assert bool((jnp.abs(grads["params"]["reducer_hidden0"]["kernel"]) > 0).any())


def test_max_pool_is_permutation_invariant():
    """PointNet's symmetric g-pool: permuting the patch tokens must not change
    the house embedding (the camera branch is unaffected by patch order)."""
    enc = _make_encoder()
    obs = _make_obs(batch=1, n_patches=64, token_dim=16)
    params = enc.init(jax.random.PRNGKey(1), obs)
    _, house_before = enc.apply(params, obs, method="branches")

    perm = jax.random.permutation(jax.random.PRNGKey(7), 64)
    obs_perm = {
        **obs,
        GLOBAL_PATCH_TOKENS_KEY: obs[GLOBAL_PATCH_TOKENS_KEY][:, perm, :],
    }
    _, house_after = enc.apply(params, obs_perm, method="branches")
    np.testing.assert_allclose(house_before, house_after, atol=1e-5)


def test_camera_token_is_isolated_from_house_branch():
    """The camera token rides its own side branch: changing it must move only
    the camera embedding, and changing a patch token must move only the house
    embedding (the deliberate split from IDEA.md)."""
    enc = _make_encoder()
    obs = _make_obs(batch=1, n_patches=64, token_dim=16)
    params = enc.init(jax.random.PRNGKey(1), obs)
    cam_before, house_before = enc.apply(params, obs, method="branches")

    # Change only the camera token.
    obs_cam = {
        **obs,
        CAMERA_TOKEN_GLOBAL_KEY: obs[CAMERA_TOKEN_GLOBAL_KEY] + 1.0,
    }
    cam_after, house_after_cam = enc.apply(params, obs_cam, method="branches")
    assert bool(jnp.any(jnp.abs(cam_after - cam_before) > 1e-4))
    np.testing.assert_allclose(house_after_cam, house_before, atol=1e-5)

    # Change only one patch token.
    patches = obs[GLOBAL_PATCH_TOKENS_KEY].at[:, 0, :].add(1.0)
    obs_patch = {**obs, GLOBAL_PATCH_TOKENS_KEY: patches}
    cam_after_patch, house_after_patch = enc.apply(
        params, obs_patch, method="branches"
    )
    np.testing.assert_allclose(cam_after_patch, cam_before, atol=1e-5)
    assert bool(jnp.any(jnp.abs(house_after_patch - house_before) > 1e-4))


def test_rejects_wrong_camera_token_shape():
    enc = _make_encoder()
    obs = _make_obs(batch=2, n_patches=64, token_dim=16)
    obs[CAMERA_TOKEN_GLOBAL_KEY] = jax.random.normal(
        jax.random.PRNGKey(9), (2, 5, 16), dtype=jnp.float32
    )  # not a singleton token axis
    with pytest.raises(ValueError, match="camera_token_global"):
        enc.init(jax.random.PRNGKey(1), obs)


def test_rejects_wrong_patch_token_count():
    enc = _make_encoder(num_patch_tokens=64)
    obs = _make_obs(batch=2, n_patches=50, token_dim=16)
    with pytest.raises(ValueError, match="global_patch_tokens"):
        enc.init(jax.random.PRNGKey(1), obs)


def test_reducer_nonlinearity_before_pool_is_present():
    """Guards the design caveat: a Dense->silu before the max-pool is required,
    otherwise two linear layers around a linear pool collapse to one Linear."""
    enc = _make_encoder()
    params = enc.init(
        jax.random.PRNGKey(1), _make_obs(batch=1, n_patches=64, token_dim=16)
    )
    # The reducer MLP and its RMSNorm must exist as trained parameters.
    assert "reducer_hidden0" in params["params"]
    assert "reducer_norm0" in params["params"]


def test_short_training_loop_is_stable():
    enc = _make_encoder()
    obs = _make_obs(batch=4, n_patches=64, token_dim=16)
    target = jax.random.normal(jax.random.PRNGKey(3), (4, 64), dtype=jnp.float32)
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