"""Value-target helpers shared by behavior and representation losses."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


class LambdaReturnInputs(NamedTuple):
    """Inputs for generalized lambda-return targets."""

    last: Any
    term: Any
    reward: Any
    value: Any
    boot: Any
    disc: Any
    lamb: Any


def lambda_return(inputs: LambdaReturnInputs):
    """Compute lambda-returns (generalized advantage estimation target).

    All inputs: (..., T, 1).
    Returns: (..., T-1, 1).
    """
    live = (1.0 - inputs.term)[..., 1:, :] * inputs.disc
    cont = (1.0 - inputs.last)[..., 1:, :] * inputs.lamb
    interm = (
        inputs.reward[..., 1:, :]
        + (1.0 - cont) * live * inputs.boot[..., 1:, :]
    )
    time_minus_one = live.shape[-2]

    def _scan_fn(carry, i):
        idx = time_minus_one - 1 - i
        val = interm[..., idx, :] + live[..., idx, :] * cont[..., idx, :] * carry
        return val, val

    init = inputs.boot[..., -1, :]
    _, outs = jax.lax.scan(_scan_fn, init, jnp.arange(time_minus_one))
    outs = jnp.flip(outs, axis=0)
    ndim = outs.ndim
    axes = list(range(1, ndim - 1)) + [0, ndim - 1]
    return jnp.transpose(outs, axes)
