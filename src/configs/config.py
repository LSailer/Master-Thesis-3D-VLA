"""Public configuration exports for R2Dreamer.

The concrete config dataclasses live in focused modules:

- ``agent_config.py``: ``R2DreamerConfig`` and agent architecture/loss knobs.
- ``agent_interface.py``: ``R2DreamerInterfaceConfig`` for obs/action shapes.
- ``trainer_config.py``: ``TrainerConfig`` and replay/training-loop knobs.
- ``observation_config.py``: observation/replay layout contracts.
"""

from src.configs.agent_config import LATENT_PRESETS, R2DreamerConfig
from src.configs.agent_interface import R2DreamerInterfaceConfig
from src.configs.observation_config import (
    ObservationDims,
    ObservationRunConfig,
    ReplayObservationConfig,
)
from src.configs.trainer_config import TrainerConfig

__all__ = [
    "LATENT_PRESETS",
    "ObservationDims",
    "ObservationRunConfig",
    "R2DreamerConfig",
    "R2DreamerInterfaceConfig",
    "ReplayObservationConfig",
    "TrainerConfig",
]
