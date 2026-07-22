"""Decoder modules paired with encoder observations for diagnostics."""

import flax.linen as nn
import jax.numpy as jnp

from src.r2dreamer.world_model.rssm import RMSNorm


class ConvDecoder(nn.Module):
    """Transpose-conv image decoder for visual verification.

    Maps an RSSM feature ``(B, F)`` back to an RGB image ``(B, 64, 64, 3)`` in
    ``[0, 1]``. The decoder is a stop-gradient visualisation probe when enabled;
    it is not part of the default world-model objective.
    """

    depth: int = 16
    kernel_size: int = 5
    mults: tuple[int, ...] = (2, 3, 4, 4)
    out_channels: int = 3
    base_res: int = 4  # 4x4 -> (x2)^4 -> 64x64

    @nn.compact
    def __call__(self, feat: jnp.ndarray) -> jnp.ndarray:
        """Decode flat RSSM features into HWC RGB images."""
        batch = feat.shape[0]
        first_channels = self.depth * self.mults[-1]
        x = nn.Dense(self.base_res * self.base_res * first_channels, name="in")(feat)
        x = x.reshape(batch, self.base_res, self.base_res, first_channels)
        for i, mult in enumerate(reversed(self.mults)):
            channels = self.depth * mult
            x = nn.ConvTranspose(
                channels,
                (self.kernel_size, self.kernel_size),
                strides=(2, 2),
                padding="SAME",
                name=f"deconv{i}",
            )(x)
            x = RMSNorm(name=f"norm{i}")(x)
            x = nn.silu(x)
        x = nn.Conv(
            self.out_channels,
            (self.kernel_size, self.kernel_size),
            padding="SAME",
            name="out",
        )(x)
        x = nn.sigmoid(x)
        return x  # NHWC, matching the HWC observation contract
