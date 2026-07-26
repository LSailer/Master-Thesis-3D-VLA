"""MLP-based encoder module for flat observations.

``RMSNorm`` is re-exported here because the sibling encoder modules and the
routed composite encoder consume it alongside :class:`MLPEncoder`.
"""

import flax.linen as nn
import jax.numpy as jnp

from src.r2dreamer.world_model.rssm import RMSNorm


class MLPEncoder(nn.Module):
    """Generic flat-feature MLP encoder.

    Parameters:
        embed_dim: Output embedding width consumed by the RSSM posterior.
        hidden: Width of each hidden block.
        num_layers: Number of hidden ``Dense -> RMSNorm -> SiLU`` blocks.
            ``num_layers=0`` is a linear encoder: only ``Dense(embed_dim)``.

    Returns:
        A float array with shape ``(..., embed_dim)``. Leading dimensions are
        preserved by Flax dense broadcasting.
    """

    embed_dim: int = 1024
    hidden: int = 1024
    num_layers: int = 1

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Encode one flat observation array into an RSSM embedding."""
        x = jnp.asarray(obs)
        for i in range(self.num_layers):
            x = nn.Dense(self.hidden, name=f"hidden{i}")(x)
            x = RMSNorm(name=f"norm{i}")(x)
            x = nn.silu(x)
        return nn.Dense(self.embed_dim, name="proj")(x)
