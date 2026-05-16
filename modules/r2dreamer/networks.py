"""Backward-compatible re-export shim.

The network modules now live in `world_model/`, `behavior/`, and
`representation/` subpackages. Existing imports of the form
``from modules.r2dreamer.networks import X`` continue to work via this file.
Prefer importing from the subpackages directly in new code.
"""

from .world_model.rssm import RMSNorm, BlockLinear, Deter, R2RSSM
from .world_model.encoders import (
    R2Encoder,
    VGGTEncoder,
    VGGTAggregatorMLPEncoder,
)
from .world_model.heads import (
    R2MLP,
    R2TwoHotDist,
    onehot_mode_st,
    _symexp,
    _make_bins,
)
from .behavior.return_ema import ReturnEMA
from .representation.barlow import Projector

__all__ = [
    "RMSNorm", "BlockLinear", "Deter", "R2RSSM",
    "R2Encoder", "VGGTEncoder", "VGGTAggregatorMLPEncoder",
    "R2MLP", "R2TwoHotDist", "onehot_mode_st",
    "Projector", "ReturnEMA",
]
