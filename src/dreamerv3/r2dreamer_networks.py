"""R2-Dreamer network modules — JAX/Flax port of NM512/r2dreamer."""

import jax
import jax.numpy as jnp
import flax.linen as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    eps: float = 1e-4

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],))
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * scale


class BlockLinear(nn.Module):
    """Block-diagonal linear layer.
    Weight layout: (out_per_block, in_per_block, blocks).
    Matches r2dreamer's einsum: "...gi,oig->...go".
    """
    out_features: int
    blocks: int = 8

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        in_features = x.shape[-1]
        in_per_block = in_features // self.blocks
        out_per_block = self.out_features // self.blocks

        kernel = self.param(
            "kernel",
            nn.initializers.lecun_normal(),
            (out_per_block, in_per_block, self.blocks),
        )
        bias = self.param("bias", nn.initializers.zeros, (self.out_features,))

        batch_shape = x.shape[:-1]
        x = x.reshape(*batch_shape, self.blocks, in_per_block)
        x = jnp.einsum("...gi,oig->...go", x, kernel)
        x = x.reshape(*batch_shape, self.out_features)
        return x + bias


class Deter(nn.Module):
    """Block-GRU deterministic state transition."""
    deter_size: int = 2048
    stoch_size: int = 512
    act_dim: int = 17
    hidden: int = 256
    blocks: int = 8
    dyn_layers: int = 1

    @nn.compact
    def __call__(self, stoch, deter, action):
        # stoch: (B, stoch_size), deter: (B, deter_size), action: (B, act_dim)

        # Normalize action magnitude
        action = action / jnp.clip(jnp.abs(action), min=1.0)

        # Three input projections
        x0 = nn.silu(RMSNorm(name="in_norm0")(nn.Dense(self.hidden, name="in0")(deter)))
        x1 = nn.silu(RMSNorm(name="in_norm1")(nn.Dense(self.hidden, name="in1")(stoch)))
        x2 = nn.silu(RMSNorm(name="in_norm2")(nn.Dense(self.hidden, name="in2")(action)))

        # Concatenate: (B, 3*hidden)
        x = jnp.concatenate([x0, x1, x2], axis=-1)

        # Broadcast across blocks: (B, blocks, 3*hidden)
        x = jnp.broadcast_to(x[:, None, :], (x.shape[0], self.blocks, x.shape[-1]))

        # Per-block deter slice: (B, blocks, deter_size//blocks)
        deter_blocked = deter.reshape(deter.shape[0], self.blocks, self.deter_size // self.blocks)

        # Combine: (B, blocks, deter/blocks + 3*hidden) -> flatten
        x = jnp.concatenate([deter_blocked, x], axis=-1)
        x = x.reshape(x.shape[0], -1)  # (B, blocks*(deter/blocks + 3*hidden))

        # Hidden layers
        for i in range(self.dyn_layers):
            x = nn.silu(RMSNorm(name=f"hid_norm{i}")(
                BlockLinear(self.deter_size, self.blocks, name=f"hid{i}")(x)))

        # GRU gates: (B, 3*deter_size)
        gates = BlockLinear(3 * self.deter_size, self.blocks, name="gru")(x)

        # Split block-wise: reshape to (B, blocks, 3*dpb), then chunk
        dpb = self.deter_size // self.blocks
        gates = gates.reshape(gates.shape[0], self.blocks, 3 * dpb)
        gate_chunks = jnp.split(gates, 3, axis=-1)  # 3x (B, blocks, dpb)
        reset = jax.nn.sigmoid(gate_chunks[0].reshape(gates.shape[0], -1))
        cand = gate_chunks[1].reshape(gates.shape[0], -1)
        update = jax.nn.sigmoid(gate_chunks[2].reshape(gates.shape[0], -1) - 1.0)

        return update * jnp.tanh(reset * cand) + (1.0 - update) * deter
