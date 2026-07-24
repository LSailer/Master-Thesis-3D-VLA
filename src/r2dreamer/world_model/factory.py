"""World-model module construction from the agent config.

Owns the cfg -> RSSM construction that used to live in
``encoders/factory.py`` (DELETIONS.md: move ``_make_rssm`` into the
world_model package before deleting the encoder factory).
"""

from __future__ import annotations

from typing import Any

from src.configs.config import R2DreamerConfig
from src.shared.dtypes import compute_jnp_dtype

from .rssm import R2RSSM


def compute_dtype_kwargs(cfg: R2DreamerConfig) -> dict[str, Any]:
    """Return the ``compute_dtype`` override for the ``full_bf16`` gate.

    Only supplies ``compute_dtype`` when ``cfg.full_bf16`` is set, so that with
    the gate off each module keeps its own default — historically float32 for
    the CNN/house/pose/RSSM/head path, but bfloat16 for modules that already
    opted in on their own (e.g. the PointNet house branch).

    Args:
      cfg: Agent config supplying ``full_bf16`` and ``compute_dtype``.

    Returns:
      ``{"compute_dtype": <jnp dtype>}`` when the gate is on, else ``{}``.
    """
    if getattr(cfg, "full_bf16", False):
        return {"compute_dtype": compute_jnp_dtype(cfg.compute_dtype)}
    return {}


def make_rssm(cfg: R2DreamerConfig) -> R2RSSM:
    """Build the RSSM module from config (incl. the full_bf16 dtype overlay)."""
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
