"""ObsAdapter: base class bridging env observations to agent/buffer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


BufferShape = tuple[int, ...] | Mapping[str, tuple[int, ...]]
BufferDType = str | Mapping[str, str]
BufferNormalize = bool | Mapping[str, bool]


@dataclass
class ObsAdapter:
    """Bridges env observations to agent/buffer, called once per step.

    Default: extracts obs["image"] for buffer (uint8), passes obs dict
    through to agent unchanged.
    """
    buffer_dtype: BufferDType = "uint8"
    buffer_shape: BufferShape = (3, 64, 64)
    normalize_on_sample: BufferNormalize = True
    agent_obs_shape: BufferShape | None = None
    on_episode_reset: Callable[[], None] | None = None

    @property
    def encoder_obs_shape(self) -> BufferShape:
        """Shape consumed by the agent encoder after any adapter/batch packing."""
        if self.agent_obs_shape is not None:
            return self.agent_obs_shape
        if isinstance(self.buffer_shape, Mapping):
            raise ValueError(
                "multi-field adapters must set agent_obs_shape for the encoder"
            )
        return self.buffer_shape

    def transform(self, obs_dict: dict) -> tuple[np.ndarray | dict[str, np.ndarray], dict]:
        """Returns (buffer_obs, agent_obs_dict)."""
        return obs_dict["image"], obs_dict

    def augment_replay_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Optionally add live adapter context to a sampled replay batch."""
        return batch
