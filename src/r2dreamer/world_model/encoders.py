"""Encoder module re-exports.

Encoder implementations live in ``src.r2dreamer.encoders``. Keep this module
thin so the world-model package no longer owns observation encoder code.
"""

from src.r2dreamer.encoders.cnn import ConvEncoder
from src.r2dreamer.encoders.constants import (
    AGG_RAW_DIM,
    AGG_REGISTER_TOKENS,
    AGG_TOKEN_TOKENS,
    HOUSE_CONTEXT_DIM,
    HYBRID_RGB_DIM,
    HYBRID_VGGT_DIM,
)
from src.r2dreamer.encoders.decoder import ConvDecoder
from src.r2dreamer.encoders.mlp import (
    HousePointsCameraEncoder,
    HybridEncoder,
    MLPEncoder,
    VGGTAggRawMLPEncoder,
    VGGTAggregatorMLPEncoder,
    WP64CNNCPMLPEncoder,
)
from src.r2dreamer.encoders.transformer import TokenTransformerEncoder

__all__ = [
    "AGG_RAW_DIM",
    "AGG_REGISTER_TOKENS",
    "AGG_TOKEN_TOKENS",
    "HOUSE_CONTEXT_DIM",
    "HYBRID_RGB_DIM",
    "HYBRID_VGGT_DIM",
    "ConvDecoder",
    "ConvEncoder",
    "HousePointsCameraEncoder",
    "HybridEncoder",
    "MLPEncoder",
    "TokenTransformerEncoder",
    "VGGTAggRawMLPEncoder",
    "VGGTAggregatorMLPEncoder",
    "WP64CNNCPMLPEncoder",
]
