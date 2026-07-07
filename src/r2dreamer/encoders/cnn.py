"""Convolutional observation encoders."""

from collections.abc import Mapping
from typing import Literal

import flax.linen as nn
import jax.numpy as jnp
from jax.typing import DTypeLike

from src.r2dreamer.encoders.shape_utils import (
    flatten_event,
    normalize_image_obs,
    restore_leading,
)
from src.r2dreamer.observation_keys import HYBRID_IMAGE_KEY, WORLD_POINTS_KEY
from src.r2dreamer.world_model.rssm import RMSNorm


def _symlog(x: jnp.ndarray) -> jnp.ndarray:
    """Symmetric log compression, ``sign(x) * log1p(|x|)``.

    Dreamer's standard transform for unbounded inputs. Used by
    ``ConvEncoder(input_kind="world_points")`` to tame the metric XYZ range of
    full-resolution world points.
    """
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


class ConvEncoder(nn.Module):
    """Shared spatial convolutional encoder for RGB images or world-point maps.

    ``input_kind`` is explicit because RGB and metric XYZ world points can share
    the same ``(B, 3, H, W)`` shape but need different preprocessing: RGB uses
    Dreamer's ``obs - 0.5`` centering, world points use symlog. ``embed_dim`` is
    optional for historical RGB compatibility; set it for world-point variants
    to project the flattened conv map to a fixed embedding width.
    """

    depth: int = 16
    kernel_size: int = 5
    mults: tuple[int, ...] = (2, 3, 4, 4)
    input_kind: Literal["rgb", "world_points"] = "rgb"
    embed_dim: int | None = None
    compute_dtype: DTypeLike = jnp.float32

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Encode ``(..., C, H, W)`` observations, preserving leading dims."""
        if isinstance(obs, Mapping):
            if self.input_kind == "world_points":
                obs = (
                    obs[WORLD_POINTS_KEY]
                    if WORLD_POINTS_KEY in obs
                    else obs["features"]
                )
            else:
                obs = obs[HYBRID_IMAGE_KEY]
        x, leading_shape = flatten_event(obs, event_ndims=3)
        if self.input_kind == "rgb":
            x = normalize_image_obs(x, dtype=self.compute_dtype) - 0.5
        elif self.input_kind == "world_points":
            x = _symlog(x.astype(self.compute_dtype))
        else:
            raise ValueError(
                f"input_kind must be 'rgb' or 'world_points', got {self.input_kind!r}"
            )
        x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        for i, mult in enumerate(self.mults):
            channels = self.depth * mult
            x = nn.Conv(
                channels,
                (self.kernel_size, self.kernel_size),
                padding="SAME",
                name=f"conv{i}",
                dtype=self.compute_dtype,
            )(x)
            x = nn.max_pool(x, (2, 2), strides=(2, 2))
            x = RMSNorm(name=f"norm{i}")(x)
            x = nn.silu(x)
        # Transpose back to NCHW before flatten to match PyTorch's flatten order.
        x = jnp.transpose(x, (0, 3, 1, 2))
        x = x.reshape(x.shape[0], -1)
        if self.embed_dim is not None:
            x = nn.Dense(self.embed_dim, name="proj", dtype=self.compute_dtype)(x)
        return restore_leading(x, leading_shape)


def make_rgb_conv_encoder(
    *,
    depth: int,
    kernel_size: int,
    mults: tuple[int, ...],
    name: str,
    compute_dtype: DTypeLike = jnp.float32,
) -> ConvEncoder:
    """Create a named RGB ``ConvEncoder`` submodule with shared defaults."""
    return ConvEncoder(
        depth=depth,
        kernel_size=kernel_size,
        mults=mults,
        name=name,
        compute_dtype=compute_dtype,
    )
