from __future__ import annotations

from typing import Callable

from src.r2dreamer.encoders import (
    Encoder,
    CNNEncoder,
    VGGTEncoder,
    VGGTAggRawMLPEncoder,
    VGGTAggregatorMLPEncoder,
    VGGTDenseWPEncoder,
    VGGTWPCP64Encoder,
    HybridAggPooledEncoder,
    HybridAggRawEncoder,
    HybridEncoder,
)
from src.r2dreamer.launch.habitat_setup import make_habitat_env


def make_crafter_env(*, seed: int = 0, **kwargs):
    """Thin wrapper so crafter follows the same factory signature as habitat."""
    from src.environments.crafter import CrafterEnv
    return CrafterEnv(size=(64, 64), seed=seed)


encoder_registry: dict[str, type[Encoder]] = {
    "cnn": CNNEncoder,
    "vggt": VGGTEncoder,
    "vggt_aggregator_mlp": VGGTAggregatorMLPEncoder,
    "vggt_agg_raw_mlp": VGGTAggRawMLPEncoder,
    "vggt_wp_dense_cnn": VGGTDenseWPEncoder,
    "vggt_wp_cp_64": VGGTWPCP64Encoder,
    "hybrid": HybridEncoder,
    "hybrid_agg_pooled": HybridAggPooledEncoder,
    "hybrid_agg_raw": HybridAggRawEncoder,
}

env_registry: dict[str, Callable] = {
    "habitat": make_habitat_env,
    "crafter": make_crafter_env,
}
