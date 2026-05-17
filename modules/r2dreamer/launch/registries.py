from __future__ import annotations

from typing import Callable

from modules.r2dreamer.encoders import (
    Encoder,
    CNNEncoder,
    VGGTEncoder,
    VGGTAggregatorMLPEncoder,
)
from modules.r2dreamer.launch.habitat_setup import make_habitat_env


def make_crafter_env(*, seed: int = 0, **kwargs):
    """Thin wrapper so crafter follows the same factory signature as habitat."""
    from modules.envs.crafter import CrafterEnv
    return CrafterEnv(size=(64, 64), seed=seed)


encoder_registry: dict[str, type[Encoder]] = {
    "cnn": CNNEncoder,
    "vggt": VGGTEncoder,
    "vggt_aggregator_mlp": VGGTAggregatorMLPEncoder,
}

env_registry: dict[str, Callable] = {
    "habitat": make_habitat_env,
    "crafter": make_crafter_env,
}
