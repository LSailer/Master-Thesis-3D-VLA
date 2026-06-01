"""Pure-CPU Flax tests for the hybrid encoder and the co-trained decoder (3D-50/51/52).

No GPU and no VGGT extractor are needed: every test operates on zeros/random
arrays of the right shape. Determinism comes from fixed ``jax.random.PRNGKey``.
"""

import jax
import jax.numpy as jnp
import pytest

from src.r2dreamer.world_model.encoders import (
    ConvDecoder,
    HybridEncoder,
    HYBRID_RGB_DIM,
    HYBRID_VGGT_DIM,
)


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


# CNN branch for the default mults (2, 3, 4, 4) at depth 16: 16*4 * 4*4 = 1024.
def _cnn_dim(depth=16, mults=(2, 3, 4, 4)):
    return depth * mults[-1] * 4 * 4


class TestHybridEncoder:
    def test_call_output_shape(self, rng):
        vggt_embed_dim = 8
        enc = HybridEncoder(
            vggt_embed_dim=vggt_embed_dim, mlp_hidden=8, mlp_layers=2
        )
        obs = jax.random.normal(rng, (4, HYBRID_RGB_DIM + HYBRID_VGGT_DIM))
        assert obs.shape == (4, 16404)

        params = enc.init(rng, obs)
        out = enc.apply(params, obs)

        cnn_dim = _cnn_dim()
        assert out.ndim == 2
        assert out.shape == (4, cnn_dim + vggt_embed_dim)

    def test_branches_gate_zero_and_vggt_zero_at_init(self, rng):
        vggt_embed_dim = 8
        enc = HybridEncoder(
            vggt_embed_dim=vggt_embed_dim, mlp_hidden=8, mlp_layers=2
        )
        obs = jax.random.normal(rng, (4, HYBRID_RGB_DIM + HYBRID_VGGT_DIM))
        params = enc.init(rng, obs)

        outputs = enc.apply(params, obs, method=enc.branches)
        assert len(outputs) == 3
        cnn_e, vggt_e, gate = outputs

        cnn_dim = _cnn_dim()
        assert cnn_e.shape == (4, cnn_dim)
        assert vggt_e.shape == (4, vggt_embed_dim)

        # Zero-init scalar gate: the VGGT branch contributes exactly nothing at init.
        assert jnp.ndim(gate) == 0
        assert float(gate) == 0.0
        assert float(jnp.max(jnp.abs(vggt_e))) == 0.0

    def test_call_concatenates_branches(self, rng):
        vggt_embed_dim = 8
        enc = HybridEncoder(
            vggt_embed_dim=vggt_embed_dim, mlp_hidden=8, mlp_layers=2
        )
        obs = jax.random.normal(rng, (4, HYBRID_RGB_DIM + HYBRID_VGGT_DIM))
        params = enc.init(rng, obs)

        fused = enc.apply(params, obs)
        cnn_e, vggt_e, _ = enc.apply(params, obs, method=enc.branches)
        expected = jnp.concatenate([cnn_e, vggt_e], axis=-1)
        assert jnp.allclose(fused, expected)


class TestConvDecoder:
    def test_output_shape_and_range(self, rng):
        dec = ConvDecoder()
        feat = jax.random.normal(rng, (4, 64))
        params = dec.init(rng, feat)
        out = dec.apply(params, feat)

        assert out.shape == (4, 3, 64, 64)
        assert float(jnp.min(out)) >= 0.0
        assert float(jnp.max(out)) <= 1.0
