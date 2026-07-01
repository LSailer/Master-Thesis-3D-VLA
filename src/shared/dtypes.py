"""Shared dtype resolution helpers."""

import jax.numpy as jnp


def compute_jnp_dtype(dtype: str):
    """Return the JAX dtype named by configuration strings."""
    if dtype == "float32":
        return jnp.float32
    if dtype in ("bfloat16", "bf16"):
        return jnp.bfloat16
    if dtype in ("float16", "fp16"):
        return jnp.float16
    raise ValueError(f"Unsupported compute_dtype={dtype!r}")
