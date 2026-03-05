"""DreamerV3 network modules in JAX/Equinox.

All modules are eqx.Module classes — pure pytrees compatible with jax.jit/vmap.
Follows NaturalDreamer's structure, rewritten for JAX.
"""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class LatentSample(NamedTuple):
    sample: jnp.ndarray  # (latent_size,) flattened one-hot
    logits: jnp.ndarray  # (latent_length, latent_classes)


def _mlp(
    key: jax.Array,
    in_size: int,
    hidden_size: int,
    out_size: int,
    num_layers: int,
) -> list:
    """Build a list of [Linear, Lambda(silu), ..., Linear] layers."""
    layers: list = []
    sizes = [in_size] + [hidden_size] * num_layers + [out_size]
    keys = jax.random.split(key, len(sizes) - 1)
    for i, (s_in, s_out) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(eqx.nn.Linear(s_in, s_out, key=keys[i]))
        if i < len(sizes) - 2:  # no activation after last layer
            layers.append(eqx.nn.Lambda(jax.nn.silu))
    return layers


def _categorical_sample(
    logits: jnp.ndarray,
    key: jax.Array,
    uniform_mix: float = 0.01,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample from categorical with uniform mixing + straight-through gradient.

    Args:
        logits: raw logits of shape (latent_length, latent_classes)
        key: PRNG key
        uniform_mix: fraction of uniform distribution to mix in

    Returns:
        (flat_sample, mixed_logits)
        flat_sample: (latent_length * latent_classes,) one-hot vectors concatenated
        mixed_logits: (latent_length, latent_classes) after uniform mixing
    """
    latent_length, latent_classes = logits.shape
    probs = jax.nn.softmax(logits, axis=-1)
    uniform = jnp.ones_like(probs) / latent_classes
    mixed_probs = (1 - uniform_mix) * probs + uniform_mix * uniform
    mixed_logits = jnp.log(mixed_probs + 1e-8)

    # Gumbel-softmax style: sample then straight-through
    indices = jax.random.categorical(key, mixed_logits, axis=-1)  # (latent_length,)
    one_hot = jax.nn.one_hot(indices, latent_classes)  # (latent_length, latent_classes)
    # Straight-through: gradient flows through softmax, forward uses one-hot
    soft = jax.nn.softmax(mixed_logits, axis=-1)
    sample = jax.lax.stop_gradient(one_hot - soft) + soft
    return sample.reshape(-1), mixed_logits


# ---------------------------------------------------------------------------
# CNN Encoder / Decoder
# ---------------------------------------------------------------------------


class ConvEncoder(eqx.Module):
    """4-layer CNN encoder: image → embedding vector."""

    layers: list
    linear: eqx.nn.Linear

    def __init__(self, obs_channels: int, embed_size: int, depth: int, *, key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.layers = [
            eqx.nn.Conv2d(obs_channels, depth * 1, kernel_size=4, stride=2, padding=1, key=k1),
            eqx.nn.Conv2d(depth * 1, depth * 2, kernel_size=4, stride=2, padding=1, key=k2),
            eqx.nn.Conv2d(depth * 2, depth * 4, kernel_size=4, stride=2, padding=1, key=k3),
            eqx.nn.Conv2d(depth * 4, depth * 8, kernel_size=4, stride=2, padding=1, key=k4),
        ]
        # After 4x stride-2 on 256x256: 16x16
        self.linear = eqx.nn.Linear(depth * 8 * 16 * 16, embed_size, key=k5)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: (C, H, W) → (embed_size,)"""
        for conv in self.layers:
            x = jax.nn.silu(conv(x))
        x = x.reshape(-1)
        return jax.nn.silu(self.linear(x))


class ConvDecoder(eqx.Module):
    """Transposed CNN decoder: state → reconstructed image."""

    linear: eqx.nn.Linear
    layers: list
    depth: int

    def __init__(
        self, state_size: int, obs_channels: int, depth: int, *, key
    ):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        self.depth = depth
        self.linear = eqx.nn.Linear(state_size, depth * 32 * 1 * 1, key=k1)
        self.layers = [
            eqx.nn.ConvTranspose2d(depth * 32, depth * 4, kernel_size=5, stride=2, key=k2),
            eqx.nn.ConvTranspose2d(depth * 4, depth * 2, kernel_size=5, stride=2, key=k3),
            eqx.nn.ConvTranspose2d(depth * 2, depth * 1, kernel_size=6, stride=2, key=k4),
            eqx.nn.ConvTranspose2d(depth * 1, obs_channels, kernel_size=6, stride=2, key=k5),
        ]

    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        """state: (state_size,) → (C, H, W)"""
        x = self.linear(state)
        x = x.reshape(self.depth * 32, 1, 1)
        for i, conv in enumerate(self.layers):
            if i < len(self.layers) - 1:
                x = jax.nn.silu(conv(x))
            else:
                x = conv(x)  # no activation on final layer
        return x


# ---------------------------------------------------------------------------
# RSSM Components
# ---------------------------------------------------------------------------


class RecurrentModel(eqx.Module):
    """GRU-based recurrence: (h, z, a) → h'."""

    linear: eqx.nn.Linear
    gru: eqx.nn.GRUCell

    def __init__(self, recurrent_size: int, latent_size: int, action_size: int, hidden_size: int, *, key):
        k1, k2 = jax.random.split(key)
        self.linear = eqx.nn.Linear(latent_size + action_size, hidden_size, key=k1)
        self.gru = eqx.nn.GRUCell(hidden_size, recurrent_size, key=k2)

    def __call__(
        self, h: jnp.ndarray, z: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """h: (recurrent_size,), z: (latent_size,), action: (action_size,) → h': (recurrent_size,)"""
        x = jax.nn.silu(self.linear(jnp.concatenate([z, action])))
        return self.gru(x, h)


class PriorNet(eqx.Module):
    """Prior: h → (z_hat, logits). Predicts latent from deterministic state only."""

    network: eqx.nn.Sequential
    latent_length: int
    latent_classes: int

    def __init__(
        self, recurrent_size: int, latent_length: int, latent_classes: int,
        hidden_size: int, num_layers: int, *, key,
    ):
        self.latent_length = latent_length
        self.latent_classes = latent_classes
        latent_size = latent_length * latent_classes
        layers = _mlp(key, recurrent_size, hidden_size, latent_size, num_layers)
        self.network = eqx.nn.Sequential(layers)

    def __call__(self, h: jnp.ndarray, *, key: jax.Array) -> LatentSample:
        raw = self.network(h)
        logits = raw.reshape(self.latent_length, self.latent_classes)
        sample, mixed_logits = _categorical_sample(logits, key)
        return LatentSample(sample, mixed_logits)


class PosteriorNet(eqx.Module):
    """Posterior: (h, embed) → (z, logits). Uses observation embedding."""

    network: eqx.nn.Sequential
    latent_length: int
    latent_classes: int

    def __init__(
        self, recurrent_size: int, embed_size: int, latent_length: int,
        latent_classes: int, hidden_size: int, num_layers: int, *, key,
    ):
        self.latent_length = latent_length
        self.latent_classes = latent_classes
        latent_size = latent_length * latent_classes
        layers = _mlp(key, recurrent_size + embed_size, hidden_size, latent_size, num_layers)
        self.network = eqx.nn.Sequential(layers)

    def __call__(
        self, h: jnp.ndarray, embed: jnp.ndarray, *, key: jax.Array
    ) -> LatentSample:
        raw = self.network(jnp.concatenate([h, embed]))
        logits = raw.reshape(self.latent_length, self.latent_classes)
        sample, mixed_logits = _categorical_sample(logits, key)
        return LatentSample(sample, mixed_logits)


# ---------------------------------------------------------------------------
# Prediction Heads
# ---------------------------------------------------------------------------


class RewardModel(eqx.Module):
    """Predict reward distribution from world model state."""

    network: eqx.nn.Sequential

    def __init__(self, state_size: int, hidden_size: int, num_layers: int, *, key):
        layers = _mlp(key, state_size, hidden_size, 2, num_layers)  # mean + logstd
        self.network = eqx.nn.Sequential(layers)

    def __call__(self, state: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Returns (mean, std) scalars."""
        out = self.network(state)
        mean, log_std = out[0], out[1]
        return mean, jnp.exp(log_std)


class ContinueModel(eqx.Module):
    """Predict episode continuation probability (Bernoulli logit)."""

    network: eqx.nn.Sequential

    def __init__(self, state_size: int, hidden_size: int, num_layers: int, *, key):
        layers = _mlp(key, state_size, hidden_size, 1, num_layers)
        self.network = eqx.nn.Sequential(layers)

    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        """Returns logit scalar."""
        return self.network(state).squeeze(-1)


class DiscreteActor(eqx.Module):
    """Discrete action policy: state → action logits."""

    network: eqx.nn.Sequential
    action_size: int

    def __init__(self, state_size: int, action_size: int, hidden_size: int, num_layers: int, *, key):
        self.action_size = action_size
        layers = _mlp(key, state_size, hidden_size, action_size, num_layers)
        self.network = eqx.nn.Sequential(layers)

    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        """Returns action logits of shape (action_size,)."""
        return self.network(state)

    def sample(self, state: jnp.ndarray, *, key: jax.Array) -> jnp.ndarray:
        """Sample action as one-hot with straight-through gradient."""
        logits = self(state)
        idx = jax.random.categorical(key, logits)
        one_hot = jax.nn.one_hot(idx, self.action_size)
        soft = jax.nn.softmax(logits)
        return jax.lax.stop_gradient(one_hot - soft) + soft


class Critic(eqx.Module):
    """Value function: state → value distribution."""

    network: eqx.nn.Sequential

    def __init__(self, state_size: int, hidden_size: int, num_layers: int, *, key):
        layers = _mlp(key, state_size, hidden_size, 2, num_layers)  # mean + logstd
        self.network = eqx.nn.Sequential(layers)

    def __call__(self, state: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Returns (mean, std) scalars."""
        out = self.network(state)
        mean, log_std = out[0], out[1]
        return mean, jnp.exp(log_std)


# ---------------------------------------------------------------------------
# Full World Model (groups all RSSM + prediction components)
# ---------------------------------------------------------------------------


class WorldModel(eqx.Module):
    encoder: ConvEncoder
    decoder: ConvDecoder
    recurrent: RecurrentModel
    prior: PriorNet
    posterior: PosteriorNet
    reward: RewardModel
    continue_model: ContinueModel

    def __init__(self, config, *, key):
        keys = jax.random.split(key, 7)
        obs_channels = config.obs_shape[0]
        embed_size = config.hidden_size

        self.encoder = ConvEncoder(obs_channels, embed_size, config.cnn_depth, key=keys[0])
        self.decoder = ConvDecoder(config.state_size, obs_channels, config.cnn_depth, key=keys[1])
        self.recurrent = RecurrentModel(
            config.recurrent_size, config.latent_size, config.action_size,
            config.hidden_size, key=keys[2],
        )
        self.prior = PriorNet(
            config.recurrent_size, config.latent_length, config.latent_classes,
            config.hidden_size, config.num_layers, key=keys[3],
        )
        self.posterior = PosteriorNet(
            config.recurrent_size, embed_size, config.latent_length,
            config.latent_classes, config.hidden_size, config.num_layers, key=keys[4],
        )
        self.reward = RewardModel(config.state_size, config.hidden_size, config.num_layers, key=keys[5])
        self.continue_model = ContinueModel(config.state_size, config.hidden_size, config.num_layers, key=keys[6])

    def initial_state(self, batch_size: int = 1) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return zeros for (h, z). If batch_size > 1, returns (batch_size, dim)."""
        if batch_size == 1:
            h = jnp.zeros(self.recurrent.gru.hidden_size)
            z = jnp.zeros(self.prior.latent_length * self.prior.latent_classes)
            return h, z
        h = jnp.zeros((batch_size, self.recurrent.gru.hidden_size))
        z = jnp.zeros((batch_size, self.prior.latent_length * self.prior.latent_classes))
        return h, z
