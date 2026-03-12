"""DreamerV3 network modules — Flax Linen."""

from typing import Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn


def symlog(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1.0)


# ---------- CNN Encoder ----------

class Encoder(nn.Module):
    depth: int = 48  # channel multiplier
    hidden_size: int = 512

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        # obs: (B, 3, H, W) float [0,1]
        x = symlog(obs)
        x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW → NHWC for Flax conv
        for i, mult in enumerate([1, 2, 4, 8]):
            x = nn.Conv(self.depth * mult, (4, 4), strides=(2, 2), name=f"conv{i}")(x)
            x = nn.silu(x)
        x = x.reshape(x.shape[0], -1)  # flatten
        x = nn.Dense(self.hidden_size, name="fc")(x)
        x = nn.silu(x)
        return x  # (B, hidden_size)


# ---------- CNN Decoder ----------

class Decoder(nn.Module):
    depth: int = 48
    hidden_size: int = 512
    obs_shape: Sequence[int] = (3, 256, 256)

    @nn.compact
    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        # features: (B, stoch_size + hidden_size)
        # Compute spatial dims after 4 stride-2 convs
        H, W = self.obs_shape[1], self.obs_shape[2]
        h, w = H // 16, W // 16
        init_channels = self.depth * 8

        x = nn.Dense(h * w * init_channels, name="fc")(features)
        x = nn.silu(x)
        x = x.reshape(x.shape[0], h, w, init_channels)  # NHWC

        for i, mult in enumerate([4, 2, 1]):
            x = nn.ConvTranspose(
                self.depth * mult, (4, 4), strides=(2, 2),
                padding="SAME", name=f"deconv{i}",
            )(x)
            x = nn.silu(x)

        x = nn.ConvTranspose(3, (4, 4), strides=(2, 2), padding="SAME", name="deconv_out")(x)
        x = jnp.transpose(x, (0, 3, 1, 2))  # NHWC → NCHW
        return x  # (B, 3, H, W)


# ---------- MLP helper ----------

class MLP(nn.Module):
    hidden: int = 512
    layers: int = 2
    out_dim: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i in range(self.layers):
            x = nn.Dense(self.hidden, name=f"fc{i}")(x)
            x = nn.silu(x)
        x = nn.Dense(self.out_dim, name="out")(x)
        return x


# ---------- RSSM ----------

class RSSM(nn.Module):
    hidden_size: int = 512
    latent_classes: int = 32
    latent_dims: int = 32
    num_actions: int = 4
    uniform_mix: float = 0.01

    @property
    def stoch_size(self) -> int:
        return self.latent_classes * self.latent_dims

    @nn.compact
    def __call__(self, h_prev, z_prev, action, embed=None):
        """Single RSSM step. Returns (h, prior_logits, post_logits, z).

        If embed is None, returns prior sample only (imagination).
        """
        # action one-hot
        a = jax.nn.one_hot(action, self.num_actions)  # (B, num_actions)
        inp = jnp.concatenate([z_prev, a], axis=-1)  # (B, stoch+actions)

        # GRU step
        inp = nn.Dense(self.hidden_size, name="gru_input")(inp)
        h, _ = nn.GRUCell(features=self.hidden_size, name="gru")(h_prev, inp)

        # Prior
        prior_logits = self._head(h, name="prior")

        # Posterior (if we have observations)
        if embed is not None:
            post_inp = jnp.concatenate([h, embed], axis=-1)
            post_logits = self._head(post_inp, name="posterior")
            z = self._sample(post_logits)
            return h, prior_logits, post_logits, z
        else:
            z = self._sample(prior_logits)
            return h, prior_logits, None, z

    def _head(self, x, name):
        x = nn.Dense(self.latent_classes * self.latent_dims, name=name)(x)
        return x.reshape(x.shape[0], self.latent_classes, self.latent_dims)

    def _sample(self, logits):
        """Categorical sample with straight-through + uniform mix."""
        # Uniform mix for exploration
        uniform = jnp.ones_like(logits) / self.latent_dims
        logits = (1 - self.uniform_mix) * logits + self.uniform_mix * jnp.log(uniform)

        # Straight-through Gumbel softmax (hard)
        soft = jax.nn.softmax(logits, axis=-1)
        hard = jax.nn.one_hot(jnp.argmax(logits, axis=-1), self.latent_dims)
        z = hard + soft - jax.lax.stop_gradient(soft)  # straight-through
        return z.reshape(z.shape[0], -1)  # (B, stoch_size)

    def initial_state(self, batch_size):
        h = jnp.zeros((batch_size, self.hidden_size))
        z = jnp.zeros((batch_size, self.stoch_size))
        return h, z


# ---------- Actor ----------

class Actor(nn.Module):
    hidden: int = 512
    layers: int = 2
    num_actions: int = 4

    @nn.compact
    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        """Returns action logits (B, num_actions)."""
        return MLP(self.hidden, self.layers, self.num_actions, name="mlp")(features)


# ---------- Critic ----------

class Critic(nn.Module):
    hidden: int = 512
    layers: int = 2

    @nn.compact
    def __call__(self, features: jnp.ndarray) -> jnp.ndarray:
        """Returns scalar value (B, 1)."""
        return MLP(self.hidden, self.layers, 1, name="mlp")(features)
