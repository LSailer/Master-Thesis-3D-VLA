"""ObsAdapter: base class bridging env observations to agent/buffer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class ObsAdapter:
    """Bridges env observations to agent/buffer, called once per step.

    Default: extracts obs["image"] for buffer (uint8), passes obs dict
    through to agent unchanged.
    """
    buffer_dtype: str = "uint8"
    buffer_shape: tuple[int, ...] = (3, 64, 64)
    normalize_on_sample: bool = True
    on_episode_reset: Callable[[], None] | None = None

    def transform(self, obs_dict: dict) -> tuple[np.ndarray, dict]:
        """Returns (buffer_obs, agent_obs_dict)."""
        return obs_dict["image"], obs_dict
