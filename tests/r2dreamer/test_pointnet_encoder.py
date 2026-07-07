"""CPU tests for the PointNet house encoder (src/r2dreamer/encoders/pointnet.py)."""

import jax
import jax.numpy as jnp
import pytest

from src.r2dreamer.encoders.pointnet import PointNetHousePointsCameraEncoder
from src.r2dreamer.observation_keys import (
    CAMERA_POSE_KEY,
    HOUSE_CONTEXT_KEY,
    HOUSE_CONTEXT_SIZE_KEY,
)


def _make_obs(batch: int, n_points: int, size: int, key=0):
    rng = jax.random.PRNGKey(key)
    k_pose, k_pts = jax.random.split(rng)
    points = jax.random.uniform(k_pts, (1, n_points, 6), dtype=jnp.float32)
    mask = (jnp.arange(n_points) < size)[None, :, None]
    return {
        CAMERA_POSE_KEY: jax.random.normal(k_pose, (batch, 9), dtype=jnp.float32),
        HOUSE_CONTEXT_KEY: jnp.where(mask, points, 0.0).astype(jnp.float16),
        HOUSE_CONTEXT_SIZE_KEY: jnp.asarray(size, dtype=jnp.int32),
    }


def _make_encoder(**overrides):
    kwargs = dict(
        embed_dim=32,
        camera_hidden=16,
        num_points=64,
        tnet_mlp=(8, 16, 32),
        tnet_fc=(16, 8),
        mlp1=(8, 8),
        mlp2=(8, 16, 32),
    )
    kwargs.update(overrides)
    return PointNetHousePointsCameraEncoder(**kwargs)


def test_forward_shape_and_finite():
    enc = _make_encoder()
    obs = _make_obs(batch=5, n_points=512, size=300)
    params = enc.init(jax.random.PRNGKey(1), obs)
    out = enc.apply(params, obs)
    assert out.shape == (5, 64)
    assert bool(jnp.isfinite(out).all())


def test_tnets_start_as_identity():
    enc = _make_encoder()
    obs = _make_obs(batch=2, n_points=256, size=200)
    params = enc.init(jax.random.PRNGKey(1), obs)
    for name, k in (("input_tnet", 3), ("feature_tnet", enc.mlp1[-1])):
        transform = params["params"][name]["transform"]
        assert bool((transform["kernel"] == 0.0).all())
        assert bool((transform["bias"].reshape(k, k) == jnp.eye(k)).all())


def test_global_feature_used_directly_when_widths_match():
    # mlp2[-1] == embed_dim: the max-pooled global feature IS the house
    # embedding — no extra projection parameters.
    enc = _make_encoder()
    obs = _make_obs(batch=2, n_points=256, size=200)
    params = enc.init(jax.random.PRNGKey(1), obs)
    assert "pointnet_house_proj" not in params["params"]

    enc_proj = _make_encoder(embed_dim=16)
    params_proj = enc_proj.init(jax.random.PRNGKey(1), obs)
    assert "pointnet_house_proj" in params_proj["params"]


def test_empty_house_zeroes_house_branch():
    enc = _make_encoder()
    obs = _make_obs(batch=3, n_points=512, size=0)
    params = enc.init(jax.random.PRNGKey(1), obs)
    camera_embed, house_embed = enc.apply(params, obs, method="branches")
    assert bool(jnp.isfinite(camera_embed).all())
    assert bool((house_embed == 0.0).all())


def test_size_below_sample_count_is_finite():
    enc = _make_encoder()
    obs = _make_obs(batch=2, n_points=512, size=7)
    params = enc.init(jax.random.PRNGKey(1), obs)
    out = enc.apply(params, obs)
    assert bool(jnp.isfinite(out).all())


def test_jit_and_grad():
    enc = _make_encoder()
    obs = _make_obs(batch=2, n_points=256, size=200)
    params = enc.init(jax.random.PRNGKey(1), obs)

    @jax.jit
    def loss_fn(p):
        return jnp.sum(enc.apply(p, obs) ** 2)

    grads = jax.grad(loss_fn)(params)
    leaves = jax.tree_util.tree_leaves(grads)
    assert leaves
    assert all(bool(jnp.isfinite(g).all()) for g in leaves)
    # The shared MLPs must receive gradient signal, and so must the T-Net
    # output layers (the zero kernel blocks the T-Nets' *internal* layers at
    # init, but the transform kernel itself gets signal and unblocks them
    # after one update).
    for key in ("pointnet_mlp1_0", "pointnet_mlp2_0"):
        g = grads["params"][key]["kernel"]
        assert bool((jnp.abs(g) > 0).any())
    for name in ("input_tnet", "feature_tnet"):
        g = grads["params"][name]["transform"]["kernel"]
        assert bool((jnp.abs(g) > 0).any())


def test_rejects_multi_house_batch():
    enc = _make_encoder()
    obs = _make_obs(batch=2, n_points=256, size=100)
    obs[HOUSE_CONTEXT_KEY] = jnp.concatenate(
        [obs[HOUSE_CONTEXT_KEY], obs[HOUSE_CONTEXT_KEY]], axis=0
    )
    with pytest.raises(ValueError, match="singleton"):
        enc.init(jax.random.PRNGKey(1), obs)


def test_paper_defaults_unchanged():
    # Guards the classic PointNet shape (arXiv:1612.00593): shared MLPs
    # (64, 64) and (64, 128, 1024), T-Net head (512, 256), 1024-d global
    # feature equal to the default embed_dim.
    enc = PointNetHousePointsCameraEncoder()
    assert enc.mlp1 == (64, 64)
    assert enc.mlp2 == (64, 128, 1024)
    assert enc.tnet_mlp == (64, 128, 1024)
    assert enc.tnet_fc == (512, 256)
    assert enc.num_points == 16384
    assert enc.mlp2[-1] == enc.embed_dim == 1024


def test_bfloat16_compute_default():
    # Compute defaults to bfloat16 (repo default); params stay float32
    # (Flax param_dtype default) and the encoder output remains finite.
    enc = _make_encoder()
    assert enc.compute_dtype == jnp.bfloat16
    obs = _make_obs(batch=2, n_points=256, size=200)
    params = enc.init(jax.random.PRNGKey(1), obs)
    assert all(
        p.dtype == jnp.float32 for p in jax.tree_util.tree_leaves(params)
    )
    out = enc.apply(params, obs)
    assert bool(jnp.isfinite(out).all())


def test_short_training_loop_is_stable():
    # A short end-to-end optimization of the full encoder must keep
    # loss/grads finite and actually reduce the loss (no explosion, no
    # collapse to NaN) — also exercises the T-Nets past their zero-grad
    # first step.
    import optax

    enc = _make_encoder()
    obs = _make_obs(batch=4, n_points=512, size=400)
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
