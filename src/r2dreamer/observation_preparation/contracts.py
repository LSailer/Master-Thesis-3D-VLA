"""Observation Preparation contracts.

Observation Preparation turns raw environment observations into the replay-buffer
observation and the one-step agent observation for a chosen input mode. The
contract records the observation forms that keep replay storage, agent acting,
and the Encoder Module input aligned.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import flax.linen as nn
import numpy as np


ObservationValue = np.ndarray | dict[str, np.ndarray]


@dataclass(frozen=True)
class ObservationField:
    """Metadata for one observation field."""

    shape: tuple[int, ...]
    dtype: str
    normalize_on_sample: bool = False


@dataclass(frozen=True)
class ObservationFormContract:
    """Shape/dtype contract for a single or structured observation form."""

    fields: ObservationField | Mapping[str, ObservationField]

    @property
    def shape(self) -> tuple[int, ...]:
        if isinstance(self.fields, Mapping):
            raise ValueError("structured observation forms do not have one shape")
        return self.fields.shape

    @property
    def dtype(self) -> str:
        if isinstance(self.fields, Mapping):
            raise ValueError("structured observation forms do not have one dtype")
        return self.fields.dtype

    @property
    def normalize_on_sample(self) -> bool:
        if isinstance(self.fields, Mapping):
            raise ValueError(
                "structured observation forms do not have one normalize flag"
            )
        return self.fields.normalize_on_sample

    def buffer_shape(self) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
        if isinstance(self.fields, Mapping):
            return {key: field.shape for key, field in self.fields.items()}
        return self.fields.shape

    def buffer_dtype(self) -> str | dict[str, str]:
        if isinstance(self.fields, Mapping):
            return {key: field.dtype for key, field in self.fields.items()}
        return self.fields.dtype

    def buffer_normalize(self) -> bool | dict[str, bool]:
        if isinstance(self.fields, Mapping):
            return {
                key: field.normalize_on_sample for key, field in self.fields.items()
            }
        return self.fields.normalize_on_sample


@dataclass(frozen=True)
class EncoderInputContract:
    """Resolved contract connecting preparation to an Encoder Module."""

    observation_preparation_type: str
    encoder_type: str
    env_render_resolution: int
    encoder_module_cls: type[nn.Module]
    env_observation: ObservationFormContract
    replay_observation: ObservationFormContract
    agent_observation: ObservationFormContract
    encoder_input: ObservationFormContract
    decoder_target: ObservationFormContract | None
    agent_overrides: Mapping[str, Any] = field(default_factory=dict)
    design_notes: str = ""


@dataclass(frozen=True)
class PreparedObservation:
    """Per-step prepared observation for replay storage and immediate acting."""

    replay_obs: ObservationValue
    agent_obs: dict[str, Any]
