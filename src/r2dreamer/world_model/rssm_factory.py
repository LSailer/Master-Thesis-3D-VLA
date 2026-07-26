"""Config -> RSSM module construction for the R2Dreamer agent.

The agent owns orchestration (params, optimizer, EMA, JIT'd steps); this module
owns the small translation from :class:`R2DreamerConfig` to the stateless
:class:`R2RSSM` Flax module, plus the ``full_bf16`` compute-dtype overlay that
the RSSM and the prediction heads share.
"""

from __future__ import annotations

from typing import Any

from src.configs.config import R2DreamerConfig
from src.shared.dtypes import compute_jnp_dtype

from .rssm import R2RSSM


def compute_dtype_kwargs(cfg: R2DreamerConfig) -> dict[str, Any]:
    """Return the ``compute_dtype`` override for the ``full_bf16`` gate.

    Only supplies ``compute_dtype`` when ``cfg.full_bf16`` is set, so that with
    the gate off each module keeps its own default - historically float32 for
    the RSSM and the prediction heads, but bfloat16 for modules that already
    opted in on their own.

    Args:
        cfg: Agent config supplying ``full_bf16`` and ``compute_dtype``.

    Returns:
        ``{"compute_dtype": <jnp dtype>}`` when the gate is on, else ``{}``.
    """
    if getattr(cfg, "full_bf16", False):
        return {"compute_dtype": compute_jnp_dtype(cfg.compute_dtype)}
    return {}


def rssm_from_config(cfg: R2DreamerConfig) -> R2RSSM:
    """Build the recurrent state-space model described by ``cfg``.

    Args:
        cfg: Agent config supplying the RSSM shape (deterministic width,
            stochastic latent shape, per-branch layer counts) and the
            ``full_bf16`` compute-dtype gate.

    Returns:
        The RSSM module, parameters not yet initialized.
    """
    return R2RSSM(
        deter_size=cfg.deter_size,
        stoch_classes=cfg.stoch_classes,
        stoch_discrete=cfg.stoch_discrete,
        num_actions=cfg.num_actions,
        hidden=cfg.hidden_size,
        blocks=cfg.blocks,
        dyn_layers=cfg.dyn_layers,
        obs_layers=cfg.obs_layers,
        img_layers=cfg.img_layers,
        unimix_ratio=cfg.unimix_ratio,
        **compute_dtype_kwargs(cfg),
    )
