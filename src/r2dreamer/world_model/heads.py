"""Generic MLP head + R2-Dreamer two-hot distribution.

`R2MLP` is the universal head module — instantiated with different `out_dim`
for the reward, continue, actor, and critic heads. `R2TwoHotDist` is the
real-space (symexp-binned) categorical used by reward and critic outputs.

Both are imported from the behavior package as well; keeping them here means
the behavior package has a small dependency on `world_model.heads`, which is
fine — actor/critic share the same head primitive as reward.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

from .rssm import RMSNorm


# ---------- TwoHot Distribution (R2-Dreamer real-space parameterization) ----------


def _symexp(x):
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def _make_bins(num_bins: int = 255):
    """Build R2-Dreamer's symexp-spaced bins in real space.

    Matches PyTorch: symexp(linspace(-20, 0, half)) mirrored.
    """
    if num_bins % 2 == 1:
        half = jnp.linspace(-20.0, 0.0, (num_bins - 1) // 2 + 1)
        half = _symexp(half)
        bins = jnp.concatenate([half, -half[:-1][::-1]])
    else:
        half = jnp.linspace(-20.0, 0.0, num_bins // 2)
        half = _symexp(half)
        bins = jnp.concatenate([half, -half[::-1]])
    return bins


class R2TwoHotDist:
    """TwoHot distribution matching R2-Dreamer's real-space parameterization.

    Unlike DreamerV3's TwoHotDist (which operates in symlog space), this uses
    bins = symexp(linspace(-20, 0, 128)) mirrored to 255 bins in real space.
    No symlog/symexp transform is applied to targets or predictions.
    """

    def __init__(self, num_bins: int = 255):
        self.num_bins = num_bins
        self.bins = _make_bins(num_bins)

    def encode(self, target: jnp.ndarray) -> jnp.ndarray:
        """Two-hot encode a scalar target in real space. target: (...)."""
        # Find below/above indices (matching PyTorch's logic)
        below = jnp.sum((self.bins <= target[..., None]).astype(jnp.int32), axis=-1) - 1
        above = self.num_bins - jnp.sum((self.bins > target[..., None]).astype(jnp.int32), axis=-1)
        below = jnp.clip(below, 0, self.num_bins - 1)
        above = jnp.clip(above, 0, self.num_bins - 1)
        equal = below == above
        dist_to_below = jnp.where(equal, 1.0, jnp.abs(self.bins[below] - target))
        dist_to_above = jnp.where(equal, 1.0, jnp.abs(self.bins[above] - target))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        return (weight_below[..., None] * jax.nn.one_hot(below, self.num_bins) +
                weight_above[..., None] * jax.nn.one_hot(above, self.num_bins))

    def loss(self, logits: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        """Cross-entropy loss. logits: (..., bins), target: (...) in real space."""
        twohot = jax.lax.stop_gradient(self.encode(target))
        log_probs = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
        return -(twohot * log_probs).sum(axis=-1)

    def pred(self, logits: jnp.ndarray) -> jnp.ndarray:
        """Expected value in real space (symmetric sum for numerical stability).

        logits: (..., bins). Returns: (..., 1) to match PyTorch's mode() shape.
        """
        probs = jax.nn.softmax(logits, axis=-1)
        n = self.num_bins
        if n % 2 == 1:
            m = (n - 1) // 2
            p1 = probs[..., :m]
            p2 = probs[..., m:m + 1]
            p3 = probs[..., m + 1:]
            b1 = self.bins[:m]
            b2 = self.bins[m:m + 1]
            b3 = self.bins[m + 1:]
            wavg = (jnp.sum(p2 * b2, axis=-1, keepdims=True) +
                    jnp.sum(p1[..., ::-1] * b1[::-1] + p3 * b3, axis=-1, keepdims=True))
        else:
            p1 = probs[..., :n // 2]
            p2 = probs[..., n // 2:]
            b1 = self.bins[:n // 2]
            b2 = self.bins[n // 2:]
            wavg = jnp.sum(p1[..., ::-1] * b1[::-1] + p2 * b2, axis=-1, keepdims=True)
        return wavg  # (..., 1) — real space, no unsquash needed


def onehot_mode_st(logits: jnp.ndarray, unimix_ratio: float = 0.0) -> jnp.ndarray:
    """Straight-through one-hot at argmax — JAX port of OneHotDist.mode.

    Forward: ``one_hot(argmax(log_probs))`` where ``log_probs`` are unimix-mixed
    log-softmax of ``logits``. Backward: identity through ``log_probs``, which
    propagates the ``log_softmax`` Jacobian back to ``logits``.

    Matches PyTorch R2-Dreamer (``distributions.py``): ``OneHotDist`` reparameterises
    ``self.logits = log(softmax(input) * (1-r) + r/K)`` in ``__init__``, then
    ``mode`` returns ``one_hot(argmax).detach() + self.logits - self.logits.detach()``.
    """
    K = logits.shape[-1]
    if unimix_ratio > 0:
        probs = jax.nn.softmax(logits, axis=-1)
        probs = (1.0 - unimix_ratio) * probs + unimix_ratio / K
        log_probs = jnp.log(probs + 1e-8)
    else:
        log_probs = jax.nn.log_softmax(logits, axis=-1)
    hard = jax.nn.one_hot(jnp.argmax(log_probs, axis=-1), K)
    return hard + log_probs - jax.lax.stop_gradient(log_probs)


class R2MLP(nn.Module):
    """MLP with RMSNorm, matching PyTorch R2-Dreamer's MLP + MLPHead."""
    hidden: int = 256
    layers: int = 2
    out_dim: int = 1
    outscale: float = 1.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i in range(self.layers):
            x = nn.Dense(self.hidden, name=f"fc{i}")(x)
            x = RMSNorm(name=f"norm{i}")(x)
            x = nn.silu(x)
        if self.outscale == 0.0:
            out_init = nn.initializers.zeros
        elif self.outscale != 1.0:
            base_init = nn.initializers.lecun_normal()
            def out_init(key, shape, dtype=jnp.float32):
                return base_init(key, shape, dtype) * self.outscale
        else:
            out_init = nn.initializers.lecun_normal()
        x = nn.Dense(self.out_dim, kernel_init=out_init, name="out")(x)
        return x
