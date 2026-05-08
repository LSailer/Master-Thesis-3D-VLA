"""Shape and integration tests for VGGT encoder + replay buffer."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from modules.r2dreamer.config import R2DreamerConfig
from modules.r2dreamer.networks import FiLMEncoder_v1, VGGTEncoder
from modules.shared.replay_buffer import VGGTReplayBuffer


FEATURE_DIM = 4116  # 37*37*3 + 9


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


class TestFiLMEncoderV1:
    """Test Vanilla FiLM encoder for VGGT world-points + pose features."""

    def test_output_shape(self):
        enc = FiLMEncoder_v1(embed_dim=1024, channels=64, hidden=128)
        rng = jax.random.PRNGKey(0)
        dummy = jnp.zeros((2, FEATURE_DIM))
        params = enc.init(rng, dummy)
        out = enc.apply(params, dummy)
        assert out.shape == (2, 1024)

    def test_film_branch_initializes_to_identity_modulation(self):
        enc = FiLMEncoder_v1(embed_dim=1024, channels=64, hidden=128)
        rng = jax.random.PRNGKey(0)
        dummy = jnp.zeros((2, FEATURE_DIM))
        params = enc.init(rng, dummy)
        film = params["params"]["pose_film"]
        assert jnp.all(film["kernel"] == 0.0)
        assert jnp.all(film["bias"] == 0.0)

    def test_gamma_noise_init_randomizes_only_delta_gamma_kernel(self):
        enc = FiLMEncoder_v1(
            embed_dim=1024,
            channels=64,
            hidden=128,
            gamma_init_std=0.01,
        )
        rng = jax.random.PRNGKey(0)
        dummy = jnp.zeros((2, FEATURE_DIM))
        params = enc.init(rng, dummy)
        film = params["params"]["pose_film"]
        gamma_kernel, beta_kernel = jnp.split(film["kernel"], 2, axis=-1)
        gamma_bias, beta_bias = jnp.split(film["bias"], 2, axis=-1)
        assert not jnp.all(gamma_kernel == 0.0)
        assert jnp.all(beta_kernel == 0.0)
        assert jnp.all(gamma_bias == 0.0)
        assert jnp.all(beta_bias == 0.0)

    def test_pose_skip_concat_expands_projection_input(self):
        enc = FiLMEncoder_v1(
            embed_dim=1024,
            channels=64,
            hidden=128,
            pose_skip=True,
        )
        rng = jax.random.PRNGKey(0)
        dummy = jnp.zeros((2, FEATURE_DIM))
        params = enc.init(rng, dummy)
        out = enc.apply(params, dummy)
        assert out.shape == (2, 1024)
        assert params["params"]["proj"]["kernel"].shape == (64 + 9, 1024)

    def test_agent_train_step_vggt_film_logs_stream_gradient_norms(self):
        from modules.r2dreamer.agent import R2DreamerAgent

        cfg = R2DreamerConfig(
            encoder_type="vggt_film_v1",
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

        expected_metrics = [
            "grad/obs_wp_norm",
            "grad/obs_pose_norm",
            "grad/obs_pose_to_wp_ratio",
            "film/gamma_minus_1_abs_mean",
            "film/gamma_actual_mean",
            "film/gamma_actual_std",
            "film/beta_abs_mean",
            "film/beta_rms",
            "actor/entropy",
            "actor/grad_norm",
            "critic/grad_norm",
            "wm/grad_norm",
            "kl/prior",
            "kl/post",
            "reward_pred_acc",
            "continue_pred_acc",
            "continue_pred_acc/done",
            "replay/is_first_rate",
        ]
        for name in expected_metrics:
            assert name in metrics
            assert np.isfinite(metrics[name])


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
