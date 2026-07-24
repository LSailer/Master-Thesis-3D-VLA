"""Deprecated delegators for the moved encoder factory (DELETIONS.md).

The cfg -> encoder-module construction moved to ``encoders/recipes.py`` (the
per-type ``EncoderRecipe`` registry) and the cfg -> RSSM construction to
``world_model/factory.py``. This module remains one migration step as a thin
delegation layer for the legacy import surface (``agent.py``); it is deleted
when the agent shim goes.
"""

from __future__ import annotations

from src.configs.config import R2DreamerConfig
from src.r2dreamer.encoders.recipes import build_encoder_module, dummy_encoder_obs
from src.r2dreamer.world_model.factory import compute_dtype_kwargs, make_rssm

# Legacy aliases (former private factory API).
_compute_dtype_kwargs = compute_dtype_kwargs
_make_rssm = make_rssm
_make_encoder = build_encoder_module
_dummy_encoder_obs = dummy_encoder_obs

__all__ = [
    "R2DreamerConfig",
    "_compute_dtype_kwargs",
    "_dummy_encoder_obs",
    "_make_encoder",
    "_make_rssm",
]
