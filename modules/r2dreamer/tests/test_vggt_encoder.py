"""Shape and integration tests for VGGT encoder + replay buffer."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from modules.r2dreamer.config import R2DreamerConfig
from modules.r2dreamer.world_model.encoders import VGGTEncoder, VGGTAggregatorMLPEncoder
from modules.shared.replay_buffer import VGGTReplayBuffer


FEATURE_DIM = 4116  # 37*37*3 + 9
POOL_DIM = 1024
POOLED_FEATURE_DIM = 3 * POOL_DIM  # [cam | mean_patches | max_patches]


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
        enc = VGGTAggregatorMLPEncoder(embed_dim=32, pool_dim=8, hidden=16)
        rng = jax.random.PRNGKey(0)
        dummy = jnp.arange(2 * 24, dtype=jnp.float32).reshape(2, 24)
        params = enc.init(rng, dummy)
        out = enc.apply(params, dummy)

        assert out.shape == (2, 32)
        assert set(params["params"].keys()) == {
            "norm_cam", "norm_mean", "norm_max", "hidden", "proj",
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
        from modules.r2dreamer.agent import R2DreamerAgent

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
        from modules.r2dreamer.agent import R2DreamerAgent

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
        from modules.r2dreamer.agent import R2DreamerAgent

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

    def test_agent_train_step_vggt(self):
        from modules.r2dreamer.agent import R2DreamerAgent

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
        from modules.r2dreamer.agent import R2DreamerAgent

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
