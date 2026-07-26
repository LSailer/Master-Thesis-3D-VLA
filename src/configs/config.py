"""Public configuration exports for R2Dreamer.

The concrete config dataclasses live in focused modules:

- ``agent_config.py``: ``R2DreamerConfig`` and agent architecture/loss knobs.
- ``trainer_config.py``: ``TrainerConfig`` and replay/training-loop knobs.
"""

from src.configs.agent_config import LATENT_PRESETS, R2DreamerConfig
from src.configs.trainer_config import TrainerConfig

__all__ = [
    "LATENT_PRESETS",
    "R2DreamerConfig",
    "TrainerConfig",
]
