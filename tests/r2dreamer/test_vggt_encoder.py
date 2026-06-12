"""Shape and integration tests for VGGT encoder + replay buffer."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.r2dreamer.config import R2DreamerConfig
from src.r2dreamer.world_model.encoders import (
    VGGTAggTokenTransformerEncoder,
    VGGTFullTokenContextTransformer,
    VGGTEncoder,
    VGGTAggregatorMLPEncoder,
    WPConvEncoder,
)
from src.r2dreamer.adapters.vggt_adapter import full_aggregator_tokens
from src.buffer.replay_buffer import VGGTReplayBuffer


FEATURE_DIM = 4116  # 37*37*3 + 9
POOL_DIM = 1024
POOLED_FEATURE_DIM = 3 * POOL_DIM  # [cam | mean_patches | max_patches]


class TestVGGTAggTokenTransformerEncoder:
    """Shape tests for the 3D-75 full-token Transformer encoder."""

    def test_output_shape_with_reduced_token_count(self):
        enc = VGGTAggTokenTransformerEncoder(
            embed_dim=64,
            token_dim=16,
            num_tokens=10,
            projection_dim=32,
            layers=2,
            heads=4,
            mlp_ratio=2,
            keep_register_tokens=True,
        )
        dummy = jnp.zeros((2, 10 * 16), dtype=jnp.float16)
        params = enc.init(jax.random.PRNGKey(0), dummy)
        out = enc.apply(params, dummy)
        assert out.shape == (2, 64)
        assert jnp.isfinite(out).all()

    def test_can_drop_register_tokens_for_future_ablation(self):
        enc = VGGTAggTokenTransformerEncoder(
            embed_dim=32,
            token_dim=8,
            num_tokens=10,
            projection_dim=16,
            layers=1,
            heads=4,
            keep_register_tokens=False,
        )
        dummy = jnp.ones((1, 80), dtype=jnp.float16)
        params = enc.init(jax.random.PRNGKey(1), dummy)
        out = enc.apply(params, dummy)
        assert out.shape == (1, 32)
        assert params["params"]["pos_embed"].shape == (1, 6, 16)

    def test_rejects_wrong_flat_dim(self):
        enc = VGGTAggTokenTransformerEncoder(
            embed_dim=32,
            token_dim=8,
            num_tokens=10,
            projection_dim=16,
            heads=4,
        )
        with pytest.raises(ValueError, match="flattened VGGT aggregator tokens"):
            enc.init(jax.random.PRNGKey(0), jnp.zeros((1, 79), dtype=jnp.float16))


class TestVGGTFullTokenContextTransformer:
    """Shape tests for the 3D-77 full-token context Transformer."""

    def test_full_token_source_shape(self):
        tokens = jnp.zeros((1374, 2048), dtype=jnp.float32)
        out = full_aggregator_tokens(
            {"aggregator_full_tokens": tokens},
            expected_shape=(1374, 2048),
        )
        assert out.shape == (1374, 2048)

    def test_output_shape_with_reduced_dims_and_no_token_projection(self):
        enc = VGGTFullTokenContextTransformer(
            context_dim=32,
            token_dim=16,
            num_tokens=10,
            layers=2,
            heads=4,
            mlp_ratio=2,
            dropout=0.0,
        )
        dummy = jnp.zeros((2, 10, 16), dtype=jnp.float32)
        params = enc.init(jax.random.PRNGKey(0), dummy, train=False)
        out = enc.apply(params, dummy, train=False)

        assert out.shape == (2, 32)
        assert "token_proj" not in params["params"]
        assert params["params"]["context_proj"]["kernel"].shape == (16, 32)

    def test_rejects_wrong_full_token_shape(self):
        enc = VGGTFullTokenContextTransformer(
            context_dim=32,
            token_dim=16,
            num_tokens=10,
            heads=4,
        )
        with pytest.raises(ValueError, match="full VGGT tokens"):
            enc.init(jax.random.PRNGKey(0), jnp.zeros((1, 10, 15), dtype=jnp.float32))


class TestVGGTAggregatorMLPEncoder:
    """Test Variant 1 VGGT aggregator encoder over adapter-pooled features."""

    def test_output_shape(self):
        enc = VGGTAggregatorMLPEncoder(embed_dim=1024)
        rng = jax.random.PRNGKey(0)
        dummy = jnp.zeros((2, POOLED_FEATURE_DIM))
        params = enc.init(rng, dummy)
        out = enc.apply(params, dummy)
        assert out.shape == (2, 1024)
        assert jnp.isfinite(out).all()

    def test_custom_embed_dim(self):
        enc = VGGTAggregatorMLPEncoder(embed_dim=512)
        rng = jax.random.PRNGKey(0)
        dummy = jnp.ones((1, POOLED_FEATURE_DIM), dtype=jnp.float32)
        params = enc.init(rng, dummy)
        out = enc.apply(params, dummy)
        assert out.shape == (1, 512)

    def test_rejects_unpooled_tokens(self):
        """The encoder no longer accepts (B, N, D) all-token input."""
        enc = VGGTAggregatorMLPEncoder(embed_dim=32)
        rng = jax.random.PRNGKey(0)
        dummy = jnp.zeros((2, 1374, 1024), dtype=jnp.float32)
        with pytest.raises(ValueError, match="VGGT pooled features"):
            enc.init(rng, dummy)

    def test_per_pool_norms_and_mlp_params(self):
        # Default depth = 1 hidden block (hidden0/norm0) + linear readout (proj).
        enc = VGGTAggregatorMLPEncoder(embed_dim=32, pool_dim=8, hidden=16)
        rng = jax.random.PRNGKey(0)
        dummy = jnp.arange(2 * 24, dtype=jnp.float32).reshape(2, 24)
        params = enc.init(rng, dummy)
        out = enc.apply(params, dummy)

        assert out.shape == (2, 32)
        assert set(params["params"].keys()) == {
            "norm_cam", "norm_mean", "norm_max", "hidden0", "norm0", "proj",
        }

    def test_num_layers_stacks_blocks(self):
        # num_layers=3 -> hidden0/1/2 + norm0/1/2 + per-pool norms + proj.
        enc = VGGTAggregatorMLPEncoder(embed_dim=32, pool_dim=8, hidden=16, num_layers=3)
        rng = jax.random.PRNGKey(0)
        dummy = jnp.zeros((2, 24), dtype=jnp.float32)
        params = enc.init(rng, dummy)
        out = enc.apply(params, dummy)
        assert out.shape == (2, 32)
        assert set(params["params"].keys()) == {
            "norm_cam", "norm_mean", "norm_max",
            "hidden0", "hidden1", "hidden2",
            "norm0", "norm1", "norm2",
            "proj",
        }


class TestVGGTEncoder:
    """Test VGGTEncoder Flax module."""

    def test_output_shape(self):
        enc = VGGTEncoder(embed_dim=1024)
        rng = jax.random.PRNGKey(0)
        dummy = jnp.zeros((2, FEATURE_DIM))
        params = enc.init(rng, dummy)
        out = enc.apply(params, dummy)
        assert out.shape == (2, 1024)

    def test_batched(self):
        enc = VGGTEncoder(embed_dim=1024)
        rng = jax.random.PRNGKey(0)
        dummy = jnp.ones((8, FEATURE_DIM)) * 0.5
        params = enc.init(rng, dummy)
        out = enc.apply(params, dummy)
        assert out.shape == (8, 1024)
        assert jnp.isfinite(out).all()

    def test_custom_embed_dim(self):
        enc = VGGTEncoder(embed_dim=512)
        rng = jax.random.PRNGKey(0)
        dummy = jnp.zeros((1, FEATURE_DIM))
        params = enc.init(rng, dummy)
        out = enc.apply(params, dummy)
        assert out.shape == (1, 512)

    def test_default_is_one_hidden_block_plus_proj(self):
        # 3D-52: default depth replaces the bare linear projection with an MLP.
        enc = VGGTEncoder(embed_dim=64, hidden=48)
        params = enc.init(jax.random.PRNGKey(0), jnp.zeros((1, FEATURE_DIM)))
        assert set(params["params"].keys()) == {"hidden0", "norm0", "proj"}

    def test_num_layers_three(self):
        enc = VGGTEncoder(embed_dim=64, hidden=48, num_layers=3)
        params = enc.init(jax.random.PRNGKey(0), jnp.zeros((2, FEATURE_DIM)))
        out = enc.apply(params, jnp.zeros((2, FEATURE_DIM)))
        assert out.shape == (2, 64)
        assert set(params["params"].keys()) == {
            "hidden0", "hidden1", "hidden2", "norm0", "norm1", "norm2", "proj",
        }

    def test_num_layers_zero_is_bare_linear(self):
        # Escape hatch: depth 0 reproduces the historical single-Dense projection.
        enc = VGGTEncoder(embed_dim=64, num_layers=0)
        params = enc.init(jax.random.PRNGKey(0), jnp.zeros((1, FEATURE_DIM)))
        assert set(params["params"].keys()) == {"proj"}


class TestWPConvEncoder:
    """Full-resolution world-point CNN encoder (3D-53)."""

    def test_output_shape_518(self):
        enc = WPConvEncoder(embed_dim=256)
        rng = jax.random.PRNGKey(0)
        # (B, 3, H, W) metric XYZ world-point map.
        dummy = jnp.zeros((2, 3, 518, 518))
        params = enc.init(rng, dummy)
        out = enc.apply(params, dummy)
        assert out.shape == (2, 256)
        assert jnp.isfinite(out).all()

    def test_handles_metric_xyz_range(self):
        # Values far outside [0, 1] (metric coords) must not blow up (symlog).
        enc = WPConvEncoder(embed_dim=32)
        rng = jax.random.PRNGKey(0)
        big = jnp.full((1, 3, 518, 518), 1e3)
        params = enc.init(rng, big)
        out = enc.apply(params, big)
        assert out.shape == (1, 32)
        assert jnp.isfinite(out).all()


class TestVGGTReplayBuffer:
    """Test VGGTReplayBuffer stores and samples correctly."""

    def test_add_and_sample(self):
        buf = VGGTReplayBuffer(capacity=1000, feature_dim=FEATURE_DIM)
        for i in range(200):
            features = np.random.randn(FEATURE_DIM).astype(np.float32)
            buf.add(features, action=i % 4, reward=0.1, done=(i % 50 == 49))
        assert buf.size == 200

        batch = buf.sample(batch_size=4, seq_len=16)
        assert batch["obs"].shape == (4, 16, FEATURE_DIM)
        assert batch["obs"].dtype == jnp.float32
        assert batch["actions"].shape == (4, 16)
        assert batch["rewards"].shape == (4, 16)
        assert batch["is_first"].shape == (4, 16)

    def test_no_normalization(self):
        """VGGT features should NOT be divided by 255."""
        buf = VGGTReplayBuffer(capacity=100, feature_dim=FEATURE_DIM)
        features = np.ones(FEATURE_DIM, dtype=np.float32) * 500.0
        buf.add(features, action=0, reward=0.0, done=False)
        # Add enough for sampling
        for _ in range(63):
            buf.add(features, action=0, reward=0.0, done=False)

        batch = buf.sample(batch_size=1, seq_len=16)
        # Values should be 500.0, not 500/255
        assert float(batch["obs"][0, 0, 0]) == pytest.approx(500.0)

    def test_is_first_at_boundaries(self):
        buf = VGGTReplayBuffer(capacity=1000, feature_dim=FEATURE_DIM)
        for i in range(100):
            features = np.zeros(FEATURE_DIM, dtype=np.float32)
            buf.add(features, action=0, reward=0.0, done=(i % 20 == 19))

        batch = buf.sample(batch_size=8, seq_len=16)
        # is_first should be 1.0 at t=0 of every sequence
        assert (batch["is_first"][:, 0] == 1.0).all()


class TestVGGTAgentInit:
    """Test R2DreamerAgent initializes with VGGT encoder."""

    def test_agent_init_vggt(self):
        from src.r2dreamer.agent import R2DreamerAgent

        cfg = R2DreamerConfig(
            encoder_type="vggt",
            obs_shape=(FEATURE_DIM,),
            num_actions=4,
        )
        rng = jax.random.PRNGKey(42)
        agent = R2DreamerAgent(cfg, rng)

        assert agent.embed_size == cfg.vggt_embed_dim
        assert "encoder" in agent.params

    def test_agent_act_vggt(self):
        from src.r2dreamer.agent import R2DreamerAgent

        cfg = R2DreamerConfig(
            encoder_type="vggt",
            obs_shape=(FEATURE_DIM,),
            num_actions=4,
        )
        rng = jax.random.PRNGKey(42)
        agent = R2DreamerAgent(cfg, rng)

        obs_dict = {
            "features": np.random.randn(FEATURE_DIM).astype(np.float32),
            "is_first": True,
        }
        rng, act_key = jax.random.split(rng)
        action = agent.act(obs_dict, act_key)
        assert 0 <= action < 4

    def test_agent_act_vggt_aggregator_mlp(self):
        from src.r2dreamer.agent import R2DreamerAgent

        cfg = R2DreamerConfig(
            encoder_type="vggt_aggregator_mlp",
            obs_shape=(POOLED_FEATURE_DIM,),
            num_actions=4,
        )
        rng = jax.random.PRNGKey(42)
        agent = R2DreamerAgent(cfg, rng)

        obs_dict = {
            "features": np.random.randn(POOLED_FEATURE_DIM).astype(np.float32),
            "is_first": True,
        }
        rng, act_key = jax.random.split(rng)
        action = agent.act(obs_dict, act_key)
        assert 0 <= action < 4

    def test_agent_act_vggt_agg_token_transformer_reduced_shape(self):
        from src.r2dreamer.agent import R2DreamerAgent

        cfg = R2DreamerConfig(
            encoder_type="vggt_agg_token_transformer",
            obs_shape=(10 * 16,),
            num_actions=4,
            vggt_token_count=10,
            vggt_token_dim=16,
            vggt_token_projection_dim=32,
            vggt_token_transformer_heads=4,
            vggt_token_transformer_layers=1,
            vggt_embed_dim=64,
        )
        rng = jax.random.PRNGKey(42)
        agent = R2DreamerAgent(cfg, rng)
        assert agent.embed_size == 64

        obs_dict = {
            "features": np.random.randn(10 * 16).astype(np.float16),
            "is_first": True,
        }
        rng, act_key = jax.random.split(rng)
        action = agent.act(obs_dict, act_key)
        assert 0 <= action < 4

    def test_agent_act_vggt_wp_dense(self):
        # Full wiring smoke for the dense-WP CNN path (3D-53). Small image keeps
        # the conv forward cheap on CPU; WPConvEncoder is resolution-agnostic.
        from src.r2dreamer.agent import R2DreamerAgent

        cfg = R2DreamerConfig(
            encoder_type="vggt_wp_dense_cnn",
            obs_shape=(3, 70, 70),
            num_actions=4,
        )
        rng = jax.random.PRNGKey(42)
        agent = R2DreamerAgent(cfg, rng)
        assert "encoder" in agent.params

        obs_dict = {
            "features": np.zeros((3, 70, 70), dtype=np.float32),
            "is_first": True,
        }
        rng, act_key = jax.random.split(rng)
        action = agent.act(obs_dict, act_key)
        assert 0 <= action < 4

    def test_mlp_layers_rejected_by_conv_encoders(self):
        # 3D-52 guard: vggt_mlp_layers must not be silently dropped by a conv
        # encoder (cnn or dense-WP). It should fail loud at agent construction.
        from src.r2dreamer.agent import R2DreamerAgent

        for enc_type, shape in (
            ("cnn", (3, 64, 64)),
            ("vggt_wp_dense_cnn", (3, 70, 70)),
        ):
            cfg = R2DreamerConfig(
                encoder_type=enc_type,
                obs_shape=shape,
                num_actions=4,
                vggt_mlp_layers=3,
            )
            with pytest.raises(ValueError, match="vggt_mlp_layers"):
                R2DreamerAgent(cfg, jax.random.PRNGKey(0))

    def test_agent_train_step_vggt(self):
        from src.r2dreamer.agent import R2DreamerAgent

        cfg = R2DreamerConfig(
            encoder_type="vggt",
            obs_shape=(FEATURE_DIM,),
            num_actions=4,
            batch_size=2,
            seq_len=8,
            imagination_horizon=3,
        )
        rng = jax.random.PRNGKey(42)
        agent = R2DreamerAgent(cfg, rng)

        B, T = cfg.batch_size, cfg.seq_len
        batch = {
            "obs": jnp.zeros((B, T, FEATURE_DIM)),
            "actions": jax.nn.one_hot(
                jnp.zeros((B, T), dtype=jnp.int32), cfg.num_actions
            ),
            "rewards": jnp.zeros((B, T)),
            "is_first": jnp.zeros((B, T)).at[:, 0].set(1.0),
            "is_last": jnp.zeros((B, T)),
            "is_terminal": jnp.zeros((B, T)),
        }
        rng, train_key = jax.random.split(rng)
        metrics = agent.train_step(batch, train_key)
        assert "total_loss" in metrics
        assert np.isfinite(metrics["total_loss"])

    def test_agent_train_step_vggt_aggregator_mlp(self):
        from src.r2dreamer.agent import R2DreamerAgent

        cfg = R2DreamerConfig(
            encoder_type="vggt_aggregator_mlp",
            obs_shape=(POOLED_FEATURE_DIM,),
            num_actions=4,
            batch_size=2,
            seq_len=8,
            imagination_horizon=3,
        )
        rng = jax.random.PRNGKey(42)
        agent = R2DreamerAgent(cfg, rng)

        B, T = cfg.batch_size, cfg.seq_len
        batch = {
            "obs": jnp.zeros((B, T, POOLED_FEATURE_DIM)),
            "actions": jax.nn.one_hot(
                jnp.zeros((B, T), dtype=jnp.int32), cfg.num_actions
            ),
            "rewards": jnp.zeros((B, T)),
            "is_first": jnp.zeros((B, T)).at[:, 0].set(1.0),
            "is_last": jnp.zeros((B, T)),
            "is_terminal": jnp.zeros((B, T)),
        }
        rng, train_key = jax.random.split(rng)
        metrics = agent.train_step(batch, train_key)
        assert "total_loss" in metrics
        assert np.isfinite(metrics["total_loss"])
