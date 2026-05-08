"""ObsAdapter: bridges env observations to agent/buffer.

The default base class is a passthrough for image-based env observations
(e.g. CNN encoder). VGGTObsAdapter overrides ``transform`` to run the
external feature extractor and ``reset`` to flush its KV cache.

Episode-boundary reset coordination is owned by StepDriver
(``modules.r2dreamer.step_driver``); the adapter just exposes a
``reset()`` method that StepDriver calls unconditionally.
"""

from __future__ import annotations

from dataclasses import dataclass

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

    def transform(self, obs_dict: dict) -> tuple[np.ndarray, dict]:
        """Returns (buffer_obs, agent_obs)."""
        return obs_dict["image"], obs_dict

    def reset(self) -> None:
        """Episode-boundary hook. Default no-op; subclasses with internal
        state (e.g. extractor caches) override."""
        return None
