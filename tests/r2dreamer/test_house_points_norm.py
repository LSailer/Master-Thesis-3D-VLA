"""CPU tests for house-branch XYZ normalization (``house_point_norm``).

Covers the symlog transform added to ``HousePointsCameraEncoder._house_embedding``:
that it compresses the metric XYZ channels, leaves RGB untouched, that ``"none"``
reproduces the raw-coordinate behavior, and that an invalid value raises.
"""

import jax
import jax.numpy as jnp
import pytest

from src.r2dreamer.encoders.mlp import HousePointsCameraEncoder
from src.r2dreamer.observation_keys import (
    CAMERA_POSE_KEY,
    HOUSE_CONTEXT_KEY,
    HOUSE_CONTEXT_SIZE_KEY,
)


def _make_obs(batch: int, n_points: int, size: int, key: int = 0):
    """Obs with large-magnitude metric XYZ (so symlog materially differs) and RGB in [0, 1]."""
    rng = jax.random.PRNGKey(key)
    k_pose, k_xyz, k_rgb = jax.random.split(rng, 3)
    xyz = jax.random.uniform(k_xyz, (1, n_points, 3), minval=-12.0, maxval=12.0)
    rgb = jax.random.uniform(k_rgb, (1, n_points, 3), minval=0.0, maxval=1.0)
    points = jnp.concatenate([xyz, rgb], axis=-1)
    mask = (jnp.arange(n_points) < size)[None, :, None]
    return {
        CAMERA_POSE_KEY: jax.random.normal(k_pose, (batch, 9), dtype=jnp.float32),
        HOUSE_CONTEXT_KEY: jnp.where(mask, points, 0.0).astype(jnp.float16),
        HOUSE_CONTEXT_SIZE_KEY: jnp.asarray(size, dtype=jnp.int32),
    }


def _make_encoder(norm: str) -> HousePointsCameraEncoder:
    return HousePointsCameraEncoder(
        embed_dim=32,
        camera_hidden=16,
        camera_layers=1,
        point_hidden=16,
        point_layers=2,
        house_point_norm=norm,
    )


def _symlog_xyz(house_context: jnp.ndarray) -> jnp.ndarray:
    """Reference: symlog channels [:3], leave [3:] untouched (float32, matching the encoder cast)."""
    x = house_context.astype(jnp.float32)
    xyz = jnp.sign(x[..., :3]) * jnp.log1p(jnp.abs(x[..., :3]))
    return jnp.concatenate([xyz, x[..., 3:]], axis=-1)


def test_symlog_equals_none_on_presymlogged_xyz():
    """symlog encoder on raw XYZ == none encoder on pre-symlogged XYZ (shared params).

    This pins down exactly what the transform does: symlog on channels [:3] and
    nothing on RGB [3:]. If RGB were also transformed, or a different function
    applied, the two outputs would diverge.
    """
    enc_symlog = _make_encoder("symlog")
    enc_none = _make_encoder("none")
    obs = _make_obs(batch=5, n_points=256, size=200)

    params = enc_symlog.init(jax.random.PRNGKey(1), obs)
    out_symlog = enc_symlog.apply(params, obs)

    ref_obs = dict(obs)
    ref_obs[HOUSE_CONTEXT_KEY] = _symlog_xyz(obs[HOUSE_CONTEXT_KEY])
    out_none = enc_none.apply(params, ref_obs)

    assert jnp.allclose(out_symlog, out_none, atol=1e-5)


def test_symlog_changes_output_versus_raw():
    """With large metric XYZ, symlog must actually change the encoding vs raw."""
    enc_symlog = _make_encoder("symlog")
    enc_none = _make_encoder("none")
    obs = _make_obs(batch=4, n_points=256, size=180)

    params = enc_symlog.init(jax.random.PRNGKey(2), obs)
    out_symlog = enc_symlog.apply(params, obs)
    out_none = enc_none.apply(params, obs)

    assert not jnp.allclose(out_symlog, out_none, atol=1e-3)
    assert bool(jnp.isfinite(out_symlog).all())


def test_empty_house_stays_finite_under_symlog():
    """size==0 zeros the house branch; symlog(0)==0 must not introduce NaN/Inf."""
    enc = _make_encoder("symlog")
    obs = _make_obs(batch=3, n_points=256, size=0)
    params = enc.init(jax.random.PRNGKey(3), obs)
    out = enc.apply(params, obs)
    assert out.shape == (3, 64)
    assert bool(jnp.isfinite(out).all())


def test_invalid_norm_raises():
    enc = _make_encoder("standardize")
    obs = _make_obs(batch=2, n_points=64, size=32)
    with pytest.raises(ValueError, match="house_point_norm"):
        enc.init(jax.random.PRNGKey(4), obs)
