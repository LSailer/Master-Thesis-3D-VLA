"""Tests for DreamerV3 JAX implementation.

Focuses on shape correctness and basic training mechanics.
Run: pytest tests/test_dreamer.py -v
"""

import jax
import jax.numpy as jnp
import pytest

from src.dreamer.config import DreamerConfig
from src.dreamer.networks import (
    ConvDecoder,
    ConvEncoder,
    Critic,
    DiscreteActor,
    RecurrentModel,
    PosteriorNet,
    PriorNet,
    RewardModel,
    ContinueModel,
    WorldModel,
)
from src.dreamer.replay_buffer import ReplayBuffer


# Use small sizes for fast tests
@pytest.fixture
def config():
    return DreamerConfig(
        recurrent_size=64,
        latent_length=4,
        latent_classes=4,
        hidden_size=64,
        num_layers=1,
        cnn_depth=8,
        obs_shape=(4, 64, 64),  # smaller images for tests
        action_size=4,
        batch_size=2,
        sequence_length=5,
        imagination_horizon=3,
    )


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


class TestConvEncoder:
    def test_output_shape(self, config, key):
        encoder = ConvEncoder(4, config.hidden_size, config.cnn_depth, key=key)
        obs = jnp.zeros(config.obs_shape)
        out = encoder(obs)
        assert out.shape == (config.hidden_size,)

    def test_different_inputs_different_outputs(self, config, key):
        encoder = ConvEncoder(4, config.hidden_size, config.cnn_depth, key=key)
        obs1 = jax.random.normal(key, config.obs_shape)
        obs2 = jax.random.normal(jax.random.PRNGKey(1), config.obs_shape)
        out1, out2 = encoder(obs1), encoder(obs2)
        assert not jnp.allclose(out1, out2)


class TestConvDecoder:
    def test_output_shape(self, config, key):
        decoder = ConvDecoder(config.state_size, 4, config.cnn_depth, key=key)
        state = jnp.zeros(config.state_size)
        out = decoder(state)
        assert out.shape[0] == 4  # channels
        # H, W should be > 0 (exact size depends on transposed conv arithmetic)
        assert out.ndim == 3


class TestRecurrentModel:
    def test_output_shape(self, config, key):
        recurrent = RecurrentModel(
            config.recurrent_size, config.latent_size,
            config.action_size, config.hidden_size, key=key,
        )
        h = jnp.zeros(config.recurrent_size)
        z = jnp.zeros(config.latent_size)
        a = jnp.zeros(config.action_size)
        h_new = recurrent(h, z, a)
        assert h_new.shape == (config.recurrent_size,)


class TestPriorPosterior:
    def test_prior_shape(self, config, key):
        prior = PriorNet(
            config.recurrent_size, config.latent_length, config.latent_classes,
            config.hidden_size, config.num_layers, key=key,
        )
        h = jnp.zeros(config.recurrent_size)
        result = prior(h, key=key)
        assert result.sample.shape == (config.latent_size,)
        assert result.logits.shape == (config.latent_length, config.latent_classes)

    def test_posterior_shape(self, config, key):
        posterior = PosteriorNet(
            config.recurrent_size, config.hidden_size,
            config.latent_length, config.latent_classes,
            config.hidden_size, config.num_layers, key=key,
        )
        h = jnp.zeros(config.recurrent_size)
        embed = jnp.zeros(config.hidden_size)
        result = posterior(h, embed, key=key)
        assert result.sample.shape == (config.latent_size,)
        assert result.logits.shape == (config.latent_length, config.latent_classes)


class TestPredictionHeads:
    def test_reward_model(self, config, key):
        reward = RewardModel(config.state_size, config.hidden_size, config.num_layers, key=key)
        state = jnp.zeros(config.state_size)
        mean, std = reward(state)
        assert mean.shape == ()
        assert std.shape == ()
        assert std > 0

    def test_continue_model(self, config, key):
        cont = ContinueModel(config.state_size, config.hidden_size, config.num_layers, key=key)
        state = jnp.zeros(config.state_size)
        logit = cont(state)
        assert logit.shape == ()

    def test_discrete_actor(self, config, key):
        actor = DiscreteActor(
            config.state_size, config.action_size,
            config.hidden_size, config.num_layers, key=key,
        )
        state = jnp.zeros(config.state_size)
        logits = actor(state)
        assert logits.shape == (config.action_size,)

    def test_actor_sample(self, config, key):
        actor = DiscreteActor(
            config.state_size, config.action_size,
            config.hidden_size, config.num_layers, key=key,
        )
        state = jnp.zeros(config.state_size)
        action = actor.sample(state, key=key)
        assert action.shape == (config.action_size,)
        # Should be approximately one-hot (straight-through)
        assert jnp.isclose(jnp.sum(action), 1.0, atol=0.1)

    def test_critic(self, config, key):
        critic = Critic(config.state_size, config.hidden_size, config.num_layers, key=key)
        state = jnp.zeros(config.state_size)
        mean, std = critic(state)
        assert mean.shape == ()
        assert std.shape == ()


class TestWorldModel:
    def test_init(self, config, key):
        wm = WorldModel(config, key=key)
        h, z = wm.initial_state(batch_size=1)
        assert h.shape == (config.recurrent_size,)
        assert z.shape == (config.latent_size,)

    def test_initial_state_batched(self, config, key):
        wm = WorldModel(config, key=key)
        h, z = wm.initial_state(batch_size=4)
        assert h.shape == (4, config.recurrent_size)
        assert z.shape == (4, config.latent_size)


class TestReplayBuffer:
    def test_add_and_len(self, config):
        buf = ReplayBuffer(config.obs_shape, config.action_size, capacity=100)
        assert len(buf) == 0
        for i in range(10):
            buf.add(
                jnp.zeros(config.obs_shape),
                jnp.zeros(config.action_size),
                0.0, False,
            )
        assert len(buf) == 10

    def test_sample_shape(self, config):
        buf = ReplayBuffer(config.obs_shape, config.action_size, capacity=100)
        for i in range(50):
            buf.add(
                jnp.ones(config.obs_shape) * i,
                jnp.zeros(config.action_size),
                float(i), False,
            )
        batch = buf.sample(batch_size=2, seq_len=5)
        assert batch["observations"].shape == (2, 5, *config.obs_shape)
        assert batch["actions"].shape == (2, 5, config.action_size)
        assert batch["rewards"].shape == (2, 5)
        assert batch["dones"].shape == (2, 5)

    def test_circular_buffer(self, config):
        buf = ReplayBuffer(config.obs_shape, config.action_size, capacity=5)
        for i in range(10):
            buf.add(jnp.zeros(config.obs_shape), jnp.zeros(config.action_size), 0.0, False)
        assert len(buf) == 5
        assert buf.full


class TestConfig:
    def test_derived_properties(self):
        cfg = DreamerConfig(latent_length=8, latent_classes=16, recurrent_size=256)
        assert cfg.latent_size == 128
        assert cfg.state_size == 384
