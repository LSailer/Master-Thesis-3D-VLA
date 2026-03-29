"""Shape tests for DreamerV3 networks — no Habitat or GPU required."""

import pytest
import jax
import jax.numpy as jnp

from modules.dreamerv3.configs import DreamerConfig
from modules.dreamerv3.networks import Encoder, Decoder, RSSM, MLP, Actor, Critic, symlog, symexp


@pytest.fixture
def cfg():
    return DreamerConfig()


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


class TestSymlog:
    def test_roundtrip(self):
        x = jnp.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        assert jnp.allclose(symexp(symlog(x)), x, atol=1e-5)


class TestEncoder:
    def test_output_shape(self, cfg, rng):
        enc = Encoder(cfg.encoder_depth, cfg.hidden_size)
        obs = jnp.zeros((2, *cfg.obs_shape))
        params = enc.init(rng, obs)
        out = enc.apply(params, obs)
        assert out.shape == (2, cfg.hidden_size)


class TestDecoder:
    def test_output_shape(self, cfg, rng):
        dec = Decoder(cfg.encoder_depth, cfg.hidden_size, cfg.obs_shape)
        feat = jnp.zeros((2, cfg.stoch_size + cfg.hidden_size))
        params = dec.init(rng, feat)
        out = dec.apply(params, feat)
        assert out.shape == (2, *cfg.obs_shape)


class TestRSSM:
    def test_prior_and_posterior(self, cfg, rng):
        rssm = RSSM(cfg.hidden_size, cfg.latent_classes, cfg.latent_dims, cfg.num_actions)
        B = 2
        h = jnp.zeros((B, cfg.hidden_size))
        z = jnp.zeros((B, cfg.stoch_size))
        a = jnp.zeros((B,), dtype=jnp.int32)
        embed = jnp.zeros((B, cfg.hidden_size))

        params = rssm.init(rng, h, z, a, embed)
        h_new, prior, post, z_new = rssm.apply(params, h, z, a, embed)

        assert h_new.shape == (B, cfg.hidden_size)
        assert z_new.shape == (B, cfg.stoch_size)
        assert prior.shape == (B, cfg.latent_classes, cfg.latent_dims)
        assert post.shape == (B, cfg.latent_classes, cfg.latent_dims)

    def test_imagination_mode(self, cfg, rng):
        rssm = RSSM(cfg.hidden_size, cfg.latent_classes, cfg.latent_dims, cfg.num_actions)
        B = 2
        h = jnp.zeros((B, cfg.hidden_size))
        z = jnp.zeros((B, cfg.stoch_size))
        a = jnp.zeros((B,), dtype=jnp.int32)

        # Init with embed to get params for posterior head too
        embed = jnp.zeros((B, cfg.hidden_size))
        params = rssm.init(rng, h, z, a, embed)

        # Call without embed (imagination)
        h_new, prior, post, z_new = rssm.apply(params, h, z, a)
        assert post is None
        assert z_new.shape == (B, cfg.stoch_size)

    def test_initial_state(self, cfg, rng):
        rssm = RSSM(cfg.hidden_size, cfg.latent_classes, cfg.latent_dims, cfg.num_actions)
        h = jnp.zeros((1, cfg.hidden_size))
        z = jnp.zeros((1, cfg.stoch_size))
        a = jnp.zeros((1,), dtype=jnp.int32)
        embed = jnp.zeros((1, cfg.hidden_size))
        params = rssm.init(rng, h, z, a, embed)

        h0, z0 = rssm.apply(params, method=rssm.initial_state, batch_size=4)
        assert h0.shape == (4, cfg.hidden_size)
        assert z0.shape == (4, cfg.stoch_size)


class TestActor:
    def test_output_shape(self, cfg, rng):
        actor = Actor(cfg.mlp_hidden, cfg.mlp_layers, cfg.num_actions)
        feat = jnp.zeros((2, cfg.stoch_size + cfg.hidden_size))
        params = actor.init(rng, feat)
        logits = actor.apply(params, feat)
        assert logits.shape == (2, cfg.num_actions)


class TestCritic:
    def test_output_shape(self, cfg, rng):
        critic = Critic(cfg.mlp_hidden, cfg.mlp_layers)
        feat = jnp.zeros((2, cfg.stoch_size + cfg.hidden_size))
        params = critic.init(rng, feat)
        value = critic.apply(params, feat)
        assert value.shape == (2, 1)


class TestMLP:
    def test_output_shape(self, rng):
        mlp = MLP(hidden=64, layers=2, out_dim=5)
        x = jnp.zeros((3, 10))
        params = mlp.init(rng, x)
        out = mlp.apply(params, x)
        assert out.shape == (3, 5)
