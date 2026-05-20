"""2D Rotary Position Embedding — pure functional JAX port.

Mirrors ``streamvggt.layers.rope.RotaryPositionEmbedding2D`` exactly:
split feature dim in half; vertical half rotated by y-positions, horizontal
half by x-positions. Each half is further split into pairs via
``rotate_features(x) = concat(-x[..., d/2:], x[..., :d/2])``.

Design choice: this module is *stateless*. The cos/sin tables are computed
by the caller once (usually at aggregator construction, since the patch
grid is fixed at 37x37 for the thesis) and passed in. This keeps attention
jit-friendly without a stateful frequency cache.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def compute_1d_rope_tables(
    dim: int,
    max_pos: int,
    frequency: float = 100.0,
    dtype=jnp.float32,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (cos, sin) tables of shape (max_pos, dim).

    Matches PyTorch reference:
        exponents = [0, 2, ..., dim-2] / dim
        inv_freq  = 1 / frequency ** exponents                # (dim/2,)
        angles    = outer(positions, inv_freq)                # (max_pos, dim/2)
        angles    = concat(angles, angles, axis=-1)           # (max_pos, dim)
        cos, sin  = cos(angles), sin(angles)
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")
    exponents = jnp.arange(0, dim, 2, dtype=jnp.float32) / dim
    inv_freq = 1.0 / (frequency ** exponents)
    positions = jnp.arange(max_pos, dtype=jnp.float32)
    angles = jnp.einsum("i,j->ij", positions, inv_freq)  # (max_pos, dim/2)
    angles = jnp.concatenate([angles, angles], axis=-1)  # (max_pos, dim)
    return jnp.cos(angles).astype(dtype), jnp.sin(angles).astype(dtype)


def _rotate_features(x: jnp.ndarray) -> jnp.ndarray:
    """Split last dim in halves, return concat(-x2, x1)."""
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([-x2, x1], axis=-1)


def _apply_1d_rope(
    tokens: jnp.ndarray,
    positions: jnp.ndarray,
    cos_table: jnp.ndarray,
    sin_table: jnp.ndarray,
) -> jnp.ndarray:
    """Apply 1D RoPE.

    Args:
        tokens:    (B, H, N, D) features to rotate.
        positions: (B, N) integer position indices.
        cos_table: (max_pos, D).
        sin_table: (max_pos, D).

    Returns:
        (B, H, N, D) rotated features.
    """
    # Index tables by positions; broadcast over heads.
    cos = cos_table[positions][:, None, :, :]  # (B, 1, N, D)
    sin = sin_table[positions][:, None, :, :]  # (B, 1, N, D)
    return (tokens * cos) + (_rotate_features(tokens) * sin)


def apply_rope_2d(
    tokens: jnp.ndarray,
    positions: jnp.ndarray,
    cos_table: jnp.ndarray,
    sin_table: jnp.ndarray,
) -> jnp.ndarray:
    """Apply 2D RoPE.

    Args:
        tokens:    (B, H, N, D) where D is even. Split into two (D/2) halves;
                   the first (vertical) rotated by y-positions, the second
                   (horizontal) by x-positions.
        positions: (B, N, 2) integer (y, x) positions.
        cos_table: (max_pos, D/2).
        sin_table: (max_pos, D/2).

    Returns:
        (B, H, N, D) rotated features.
    """
    if tokens.shape[-1] % 2 != 0:
        raise ValueError(f"feature dim must be even, got {tokens.shape[-1]}")
    vertical, horizontal = jnp.split(tokens, 2, axis=-1)
    vertical = _apply_1d_rope(vertical, positions[..., 0], cos_table, sin_table)
    horizontal = _apply_1d_rope(horizontal, positions[..., 1], cos_table, sin_table)
    return jnp.concatenate([vertical, horizontal], axis=-1)
