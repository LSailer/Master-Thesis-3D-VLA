"""Barlow Twins representation loss + the linear projector it uses.

Barlow Twins decorrelates a projection of the world-model feature `feat`
against the encoder embedding `embed`. The diagonal of the cross-correlation
is pushed to 1 (invariance), the off-diagonal to 0 (redundancy reduction).

`cfg.barlow_stop_grad` controls whether the gradient on `embed` reaches the
encoder (Protocol D toggle); `True` matches the original PyTorch behaviour.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn


class Projector(nn.Module):
    """Single linear projection without bias (maps feat_size -> embed_dim)."""

    out_dim: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.out_dim, use_bias=False, name="proj")(x)


def barlow_loss(*, feat, embed, params, modules, cfg, B, T):
    """Barlow Twins cross-correlation loss between projected feat and embed.

    Args:
        feat: (B, T, F) — RSSM features (gradients flow to RSSM/encoder).
        embed: (B, T, E) — encoder output (`cfg.barlow_stop_grad` controls grad).
        params: agent params (uses `projector`).
        modules: Flax module dict (uses `projector`).
        cfg: R2DreamerConfig (uses `barlow_stop_grad`, `barlow_lambda`).
        B, T: batch and time dims.

    Returns:
        scalar loss = invariance + cfg.barlow_lambda * redundancy.
    """
    feat_flat = feat.reshape(B * T, -1)
    embed_flat = embed.reshape(B * T, -1)

    x1 = modules["projector"].apply(params["projector"], feat_flat)  # (BT, E)
    x2 = jax.lax.stop_gradient(embed_flat) if cfg.barlow_stop_grad else embed_flat

    # ddof=1 matches torch.std() default (Bessel correction)
    x1_norm = (x1 - jnp.mean(x1, axis=0)) / (jnp.std(x1, axis=0, ddof=1) + 1e-8)
    x2_norm = (x2 - jnp.mean(x2, axis=0)) / (jnp.std(x2, axis=0, ddof=1) + 1e-8)

    c = (x1_norm.T @ x2_norm) / (B * T)  # (E, E)
    invariance = jnp.sum((jnp.diag(c) - 1.0) ** 2)
    off_diag = 1.0 - jnp.eye(c.shape[0])
    redundancy = jnp.sum((c * off_diag) ** 2)
    return invariance + cfg.barlow_lambda * redundancy
