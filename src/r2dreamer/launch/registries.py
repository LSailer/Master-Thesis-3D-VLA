"""Launch-time registries for R2Dreamer encoders and environments."""

from __future__ import annotations

from typing import Callable

from src.environments.crafter import CrafterEnv
from src.r2dreamer.encoders import (
    Encoder,
    CNNEncoder,
    GnnEdgeHousePointsPoseEncoder,
    GnnHousePointsPoseEncoder,
    VGGTAggTokenTransformerEncoder,
    VGGTEncoder,
    VGGTAggRawEncoder,
    VGGTAggregatorMLPEncoder,
    VGGTDenseWPEncoder,
    VGGTWPCP64Encoder,
    VGGTWP64CNNCPMLPEncoder,
    HybridEncoder,
    VGGTHouseContextEncoder,
    VGGTHouseFullTokenNoGateEncoder,
    VGGTHouseGlobalEmbeddingEncoder,
    VGGTHouseGlobalTokenNoGateEncoder,
    VGGTHousePointsPoseEncoder,
    VGGTHybridHousePointsPoseEncoder,
)
from src.r2dreamer.launch.habitat_setup import make_habitat_env


def make_crafter_env(*, seed: int = 0, **_kwargs):
    """Thin wrapper so crafter follows the same factory signature as habitat."""
    return CrafterEnv(size=(64, 64), seed=seed)


encoder_registry: dict[str, type[Encoder]] = {
    "cnn": CNNEncoder,
    "vggt": VGGTEncoder,
    "vggt_aggregator_mlp": VGGTAggregatorMLPEncoder,
    "vggt_agg_raw": VGGTAggRawEncoder,
    "vggt_agg_token_transformer": VGGTAggTokenTransformerEncoder,
    "vggt_wp_dense_cnn": VGGTDenseWPEncoder,
    "vggt_wp_cp_64": VGGTWPCP64Encoder,
    "vggt_wp64_cnn_cp_mlp": VGGTWP64CNNCPMLPEncoder,
    "hybrid": HybridEncoder,
    "vggt_house_context": VGGTHouseContextEncoder,
    "vggt_house_points_pose": VGGTHousePointsPoseEncoder,
    "vggt_hybrid_house_points_pose": VGGTHybridHousePointsPoseEncoder,
    "gnn_house_points_pose": GnnHousePointsPoseEncoder,
    "gnn_edge_house_points_pose": GnnEdgeHousePointsPoseEncoder,
    "vggt_house_full_tokens_nogate": VGGTHouseFullTokenNoGateEncoder,
    "vggt_house_global_tokens_nogate": VGGTHouseGlobalTokenNoGateEncoder,
    "vggt_house_global_embedding": VGGTHouseGlobalEmbeddingEncoder,
}

env_registry: dict[str, Callable] = {
    "habitat": make_habitat_env,
    "crafter": make_crafter_env,
}
