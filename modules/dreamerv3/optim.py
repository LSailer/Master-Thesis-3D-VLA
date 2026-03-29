"""LaProp optimizer and Adaptive Gradient Clipping (AGC) for JAX/Optax."""

from typing import NamedTuple
import jax
import jax.numpy as jnp
import optax


class LaPropState(NamedTuple):
    count: jnp.ndarray
    exp_avg: optax.Updates
    exp_avg_sq: optax.Updates
    exp_avg_lr1: jnp.ndarray
    exp_avg_lr2: jnp.ndarray


def laprop(lr=4e-4, b1=0.9, b2=0.999, eps=1e-15):
    def init_fn(params):
        return LaPropState(
            count=jnp.zeros([], jnp.int32),
            exp_avg=jax.tree.map(jnp.zeros_like, params),
            exp_avg_sq=jax.tree.map(jnp.zeros_like, params),
            exp_avg_lr1=jnp.zeros([]),
            exp_avg_lr2=jnp.zeros([]),
        )

    def update_fn(updates, state, params=None):
        count = state.count + 1

        # Second moment
        exp_avg_sq = jax.tree.map(
            lambda v, g: b2 * v + (1 - b2) * g ** 2, state.exp_avg_sq, updates)

        # LR tracking for bias correction
        exp_avg_lr1 = state.exp_avg_lr1 * b1 + (1 - b1) * lr
        exp_avg_lr2 = state.exp_avg_lr2 * b2 + (1 - b2)

        # step_size = lr / exp_avg_lr1
        bias_correction1 = exp_avg_lr1 / (lr + 1e-30)
        step_size = 1.0 / jnp.maximum(bias_correction1, 1e-30)

        # Normalize gradient: g / (sqrt(v/bc2) + eps)
        denom = jax.tree.map(
            lambda v: jnp.sqrt(v / jnp.maximum(exp_avg_lr2, 1e-30)) + eps,
            exp_avg_sq)
        normalized = jax.tree.map(lambda g, d: g / d, updates, denom)

        # First moment of normalized gradient (scaled by lr)
        exp_avg = jax.tree.map(
            lambda m, ng: b1 * m + (1 - b1) * lr * ng,
            state.exp_avg, normalized)

        # Final update: -step_size * exp_avg
        final = jax.tree.map(lambda m: -step_size * m, exp_avg)

        return final, LaPropState(count, exp_avg, exp_avg_sq, exp_avg_lr1, exp_avg_lr2)

    return optax.GradientTransformation(init_fn, update_fn)


def agc(grads, params, clip=0.3, pmin=1e-3):
    def clip_fn(g, p):
        p_norm = jnp.maximum(jnp.sqrt(jnp.sum(p ** 2)), pmin)
        g_norm = jnp.sqrt(jnp.sum(g ** 2))
        max_norm = clip * p_norm
        scale = max_norm / jnp.maximum(g_norm, 1e-8)
        return jnp.where(g_norm > max_norm, g * scale, g)
    return jax.tree.map(clip_fn, grads, params)
