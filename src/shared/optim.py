"""LaProp optimizer and Adaptive Gradient Clipping (AGC) for JAX/Optax."""

from typing import NamedTuple
import jax
import jax.numpy as jnp
import optax


class LaPropState(NamedTuple):
    """Optimizer state for the LaProp adaptive learning-rate rule.

    Attributes:
        step: Scalar step counter.
        exp_avg: First moment of bias-corrected, normalized gradients.
        exp_avg_sq: Second moment of raw gradients.
        exp_avg_lr1: Exponential average of the effective learning rate.
        exp_avg_lr2: Exponential average used for second-moment bias correction.
    """

    step: jnp.ndarray
    exp_avg: optax.Updates
    exp_avg_sq: optax.Updates
    exp_avg_lr1: jnp.ndarray
    exp_avg_lr2: jnp.ndarray


def laprop(lr=4e-4, b1=0.9, b2=0.999, eps=1e-15, warmup=0):
    """Build an Optax LaProp gradient transformation with optional warmup.

    Args:
        lr: Base learning rate after warmup completes.
        b1: Exponential decay rate for the first moment.
        b2: Exponential decay rate for the second moment.
        eps: Denominator stabilizer for normalized gradients.
        warmup: Number of steps to linearly ramp the learning rate from 0 to
            ``lr``. When 0, warmup is disabled.

    Returns:
        An Optax ``GradientTransformation`` implementing the LaProp update rule.
    """

    def init_fn(params):
        """Initialize LaProp optimizer state for ``params``."""
        return LaPropState(
            step=jnp.zeros([], jnp.int32),
            exp_avg=jax.tree.map(jnp.zeros_like, params),
            exp_avg_sq=jax.tree.map(jnp.zeros_like, params),
            exp_avg_lr1=jnp.zeros([]),
            exp_avg_lr2=jnp.zeros([]),
        )

    def update_fn(updates, state, params=None):
        """Apply one LaProp update to ``updates`` and return new state."""
        del params  # Required by Optax signature; LaProp does not use params.
        step = state.step + 1

        # Linear warmup: scale effective LR from 0 to lr over `warmup` steps
        # Matches PyTorch: LambdaLR(optimizer, lambda step: min(1, (step+1)/warmup))
        warmup_factor = jnp.where(
            warmup > 0,
            jnp.minimum(1.0, step.astype(jnp.float32) / warmup),
            1.0,
        )
        effective_lr = lr * warmup_factor

        # Second moment
        exp_avg_sq = jax.tree.map(
            lambda v, g: b2 * v + (1 - b2) * g**2, state.exp_avg_sq, updates
        )

        # LR tracking for bias correction
        exp_avg_lr1 = state.exp_avg_lr1 * b1 + (1 - b1) * effective_lr
        exp_avg_lr2 = state.exp_avg_lr2 * b2 + (1 - b2)

        # step_size = effective_lr / exp_avg_lr1
        bias_correction1 = exp_avg_lr1 / (effective_lr + 1e-30)
        step_size = 1.0 / jnp.maximum(bias_correction1, 1e-30)

        # Normalize gradient: g / (sqrt(v/bc2) + eps)
        denom = jax.tree.map(
            lambda v: jnp.sqrt(v / jnp.maximum(exp_avg_lr2, 1e-30)) + eps, exp_avg_sq
        )
        normalized = jax.tree.map(lambda g, d: g / d, updates, denom)

        # First moment of normalized gradient (scaled by effective_lr)
        exp_avg = jax.tree.map(
            lambda m, ng: b1 * m + (1 - b1) * effective_lr * ng,
            state.exp_avg,
            normalized,
        )

        # Final update: -step_size * exp_avg
        final = jax.tree.map(lambda m: -step_size * m, exp_avg)

        return final, LaPropState(step, exp_avg, exp_avg_sq, exp_avg_lr1, exp_avg_lr2)

    return optax.GradientTransformation(init_fn, update_fn)


def agc(grads, params, clip=0.3, pmin=1e-3):
    """Apply adaptive gradient clipping relative to parameter norms.

    Args:
        grads: Gradient pytree matching ``params``.
        params: Parameter pytree used to compute per-leaf norm caps.
        clip: Maximum allowed gradient norm as a fraction of parameter norm.
        pmin: Lower bound on parameter norm to avoid division by zero.

    Returns:
        A gradient pytree with each leaf clipped when its norm exceeds
        ``clip * max(||params||, pmin)``.
    """

    def clip_fn(g, p):
        """Clip one gradient leaf against its matching parameter norm."""
        p_norm = jnp.maximum(jnp.sqrt(jnp.sum(p**2)), pmin)
        g_norm = jnp.sqrt(jnp.sum(g**2))
        max_norm = clip * p_norm
        scale = max_norm / jnp.maximum(g_norm, 1e-8)
        return jnp.where(g_norm > max_norm, g * scale, g)

    return jax.tree.map(clip_fn, grads, params)
