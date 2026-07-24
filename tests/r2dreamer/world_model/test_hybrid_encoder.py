"""Pure-CPU Flax tests for the hybrid encoder and debug decoder probe (3D-50/51/52).

No GPU and no VGGT extractor are needed: every test operates on zeros/random
arrays of the right shape. Determinism comes from fixed ``jax.random.PRNGKey``.
"""

import jax
import jax.numpy as jnp
import pytest

from src.r2dreamer.encoders.constants import HYBRID_RGB_DIM, HYBRID_VGGT_DIM
from src.r2dreamer.encoders.decoder import ConvDecoder
from src.r2dreamer.encoders.mlp import (
    HousePointsCameraEncoder,
    HybridEncoder,
    HybridHousePointsCameraEncoder,
)
from src.r2dreamer.encoders.transformer import TokenTransformerEncoder
from src.r2dreamer.observation_keys import (
    CAMERA_POSE_KEY,
    HOUSE_CONTEXT_KEY,
    HYBRID_IMAGE_KEY,
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
        enc = HybridEncoder(vggt_embed_dim=vggt_embed_dim, mlp_hidden=8, mlp_layers=2)
        obs = jax.random.normal(rng, (4, HYBRID_RGB_DIM + HYBRID_VGGT_DIM))
        assert obs.shape == (4, 16404)

        params = enc.init(rng, obs)
        out = enc.apply(params, obs)

        cnn_dim = _cnn_dim()
        assert out.ndim == 2
        assert out.shape == (4, cnn_dim + vggt_embed_dim)

    def test_branches_gate_zero_and_vggt_zero_at_init(self, rng):
        vggt_embed_dim = 8
        enc = HybridEncoder(vggt_embed_dim=vggt_embed_dim, mlp_hidden=8, mlp_layers=2)
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
        enc = HybridEncoder(vggt_embed_dim=vggt_embed_dim, mlp_hidden=8, mlp_layers=2)
        obs = jax.random.normal(rng, (4, HYBRID_RGB_DIM + HYBRID_VGGT_DIM))
        params = enc.init(rng, obs)

        fused = enc.apply(params, obs)
        cnn_e, vggt_e, _ = enc.apply(params, obs, method=enc.branches)
        expected = jnp.concatenate([cnn_e, vggt_e], axis=-1)
        assert jnp.allclose(fused, expected)

    def test_house_context_width_fuses_1024_plus_1024_to_2048(self, rng):
        enc = HybridEncoder(vggt_dim=1024, vggt_embed_dim=1024)
        obs = jnp.zeros((2, HYBRID_RGB_DIM + 1024), dtype=jnp.float32)
        params = enc.init(rng, obs)

        fused = enc.apply(params, obs)
        cnn_e, vggt_e, _ = enc.apply(params, obs, method=enc.branches)

        assert cnn_e.shape == (2, 1024)
        assert vggt_e.shape == (2, 1024)
        assert fused.shape == (2, 2048)


class TestHousePointsCameraEncoder:
    def test_singleton_house_points_broadcast_over_camera_poses(self, rng):
        enc = HousePointsCameraEncoder(
            embed_dim=8,
            camera_hidden=8,
            camera_layers=1,
            point_hidden=8,
            point_layers=1,
        )
        obs = {
            CAMERA_POSE_KEY: jnp.zeros((3, 9), dtype=jnp.float16),
            HOUSE_CONTEXT_KEY: jnp.ones((1, 5, 6), dtype=jnp.float16),
        }

        params = enc.init(rng, obs)
        fused = enc.apply(params, obs)
        camera_embed, house_embed = enc.apply(params, obs, method=enc.branches)

        assert camera_embed.shape == (3, 8)
        assert house_embed.shape == (3, 8)
        assert fused.shape == (3, 16)
        assert jnp.allclose(house_embed[0], house_embed[1])
        assert jnp.allclose(house_embed[0], house_embed[2])


class TestHybridHousePointsCameraEncoder:
    def _make(self):
        return HybridHousePointsCameraEncoder(
            embed_dim=8,
            camera_hidden=8,
            camera_layers=1,
            point_hidden=8,
            point_layers=1,
            cnn_depth=2,
            cnn_kernel=3,
            cnn_mults=(1, 1),
        )

    def _obs(self, rng, batch=3):
        k1, k2 = jax.random.split(rng)
        return {
            HYBRID_IMAGE_KEY: jax.random.uniform(k1, (batch, 64, 64, 3)),
            CAMERA_POSE_KEY: jax.random.normal(k2, (batch, 9)).astype(jnp.float16),
            HOUSE_CONTEXT_KEY: jnp.ones((1, 5, 6), dtype=jnp.float16),
        }

    def test_output_is_cnn_plus_two_gated_branches(self, rng):
        enc = self._make()
        obs = self._obs(rng)
        params = enc.init(rng, obs)

        fused = enc.apply(params, obs)
        cnn_e, cam_e, house_e, _, _ = enc.apply(params, obs, method=enc.branches)

        cnn_dim = _cnn_dim(depth=2, mults=(1, 1))
        assert cnn_e.shape == (3, cnn_dim * 16)  # 2 stages: 64 -> 16 spatial
        assert cam_e.shape == (3, 8)
        assert house_e.shape == (3, 8)
        assert fused.shape == (3, cnn_e.shape[-1] + 16)
        assert jnp.allclose(fused, jnp.concatenate([cnn_e, cam_e, house_e], axis=-1))

    def test_gates_zero_at_init_so_output_equals_cnn_baseline(self, rng):
        enc = self._make()
        obs = self._obs(rng)
        params = enc.init(rng, obs)

        cnn_e, cam_e, house_e, gate_cam, gate_house = enc.apply(
            params, obs, method=enc.branches
        )

        assert float(gate_cam) == 0.0
        assert float(gate_house) == 0.0
        assert float(jnp.max(jnp.abs(cam_e))) == 0.0
        assert float(jnp.max(jnp.abs(house_e))) == 0.0
        assert float(jnp.max(jnp.abs(cnn_e))) > 0.0

        # Pose/house content is invisible at init: only the image matters.
        obs_other = dict(obs)
        obs_other[CAMERA_POSE_KEY] = obs[CAMERA_POSE_KEY] + 1.0
        obs_other[HOUSE_CONTEXT_KEY] = obs[HOUSE_CONTEXT_KEY] * 2.0
        assert jnp.allclose(enc.apply(params, obs), enc.apply(params, obs_other))

    def test_singleton_house_cloud_broadcasts_over_batch(self, rng):
        enc = self._make()
        obs = self._obs(rng)
        params = enc.init(rng, obs)
        # Force the house gate open so the branch is observable.
        params = jax.tree_util.tree_map(lambda x: x, params)
        params["params"]["gate_house"] = jnp.ones(())

        _, _, house_e, _, _ = enc.apply(params, obs, method=enc.branches)
        assert house_e.shape == (3, 8)
        assert jnp.allclose(house_e[0], house_e[1])
        assert jnp.allclose(house_e[0], house_e[2])


class TestRGBGlobalTokenTransformerEncoder:
    def test_singleton_global_tokens_are_encoded_once_and_broadcast(self, rng):
        enc = TokenTransformerEncoder(
            cnn_depth=2,
            cnn_kernel=3,
            cnn_mults=(1, 1),
            embed_dim=8,
            token_dim=8,
            num_tokens=6,
            layers=1,
            heads=2,
            mlp_ratio=2,
            token_key="global_tokens",
            image_key="image",
            singleton_tokens=True,
        )
        obs = {
            "image": jnp.zeros((3, 64, 64, 3), dtype=jnp.float32),
            "global_tokens": jnp.ones((1, 6, 8), dtype=jnp.float32),
        }

        params = enc.init(rng, obs)
        fused = enc.apply(params, obs)
        cnn_e, token_e = enc.apply(params, obs, method=enc.branches)

        assert cnn_e.shape[0] == 3
        assert token_e.shape == (3, 8)
        assert fused.shape == (3, cnn_e.shape[-1] + token_e.shape[-1])
        assert jnp.allclose(token_e[0], token_e[1])
        assert jnp.allclose(token_e[0], token_e[2])
        assert "gate" not in params["params"]

    def test_global_token_branch_is_not_zero_gated_at_init(self, rng):
        enc = TokenTransformerEncoder(
            cnn_depth=2,
            cnn_kernel=3,
            cnn_mults=(1, 1),
            embed_dim=8,
            token_dim=8,
            num_tokens=6,
            layers=1,
            heads=2,
            mlp_ratio=2,
            token_key="global_tokens",
            image_key="image",
            singleton_tokens=True,
        )
        image = jnp.zeros((2, 64, 64, 3), dtype=jnp.float32)
        obs_a = {
            "image": image,
            "global_tokens": jnp.zeros((1, 6, 8), dtype=jnp.float32),
        }
        obs_b = {
            "image": image,
            "global_tokens": jnp.ones((1, 6, 8), dtype=jnp.float32),
        }

        params = enc.init(rng, obs_a)
        _, token_a = enc.apply(params, obs_a, method=enc.branches)
        _, token_b = enc.apply(params, obs_b, method=enc.branches)

        assert not jnp.allclose(token_a, token_b)
        assert float(jnp.max(jnp.abs(token_b))) > 0.0


class TestDecoderGuard:
    """decoder=True requires an RGB modality (cnn or hybrid); else fail fast."""

    def _cfg(self, encoder_type, obs_shape):
        from src.configs.config import R2DreamerConfig

        return R2DreamerConfig(
            encoder_type=encoder_type,
            obs_shape=obs_shape,
            decoder=True,
            deter_size=64,
            stoch_classes=4,
            stoch_discrete=4,
            hidden_size=32,
            mlp_units=32,
            vggt_embed_dim=8,
            mlp_vggt_hidden=8,
            mlp_vggt_layers=2,
            num_actions=4,
        )

    def test_vggt_plus_decoder_raises(self):
        from src.r2dreamer.composition import make_learner

        with pytest.raises(ValueError, match="decoder=True requires"):
            make_learner(self._cfg("vggt", (4116,)), jax.random.PRNGKey(0))

    def test_cnn_and_hybrid_plus_decoder_build(self):
        from src.r2dreamer.composition import make_learner

        # Both RGB-bearing encoders must construct with a decoder.
        a = make_learner(self._cfg("cnn", (64, 64, 3)), jax.random.PRNGKey(0))
        assert "decoder" in a.params
        b = make_learner(self._cfg("hybrid", (16404,)), jax.random.PRNGKey(0))
        assert "decoder" in b.params
        cfg = self._cfg("vggt_house_context", (13312,))
        cfg.vggt_feature_dim = 1024
        c = make_learner(cfg, jax.random.PRNGKey(0))
        assert "decoder" in c.params

    def test_hybrid_split_mismatch_raises_value_error(self):
        from src.r2dreamer.composition import make_learner

        cfg = self._cfg("hybrid", (HYBRID_RGB_DIM + HYBRID_VGGT_DIM,))
        cfg.vggt_feature_dim = HYBRID_VGGT_DIM + 1

        with pytest.raises(ValueError, match="hybrid obs_shape/split mismatch"):
            make_learner(cfg, jax.random.PRNGKey(0))


class TestConvDecoder:
    def test_output_shape_and_range(self, rng):
        dec = ConvDecoder()
        feat = jax.random.normal(rng, (4, 64))
        params = dec.init(rng, feat)
        out = dec.apply(params, feat)

        assert out.shape == (4, 64, 64, 3)
        assert float(jnp.min(out)) >= 0.0
        assert float(jnp.max(out)) <= 1.0
