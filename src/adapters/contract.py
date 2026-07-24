"""Adapter contract for the multi-episode house-context prototype.

One transform call returns a flat list of routed fields. Each field says
which encoder branch consumes it (``encoder``) and whether it is stored in
replay (``buffer``). ``is_first`` is NOT part of the transform output - the
collector passes it explicitly when appending to the buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol

import jax.numpy as jnp
import numpy as np

from src.buffer.replay_buffer import ReplayBatch, ReplayTransition
from src.environments.observation import ObservationFrame
from src.vggt.jax.feature_extractor import VGGTExtractOutput


class Encoder(Enum):
    """Trainable encoder branch that consumes a field live."""

    CONV = auto()
    MLP = auto()
    POINTNET = auto()
    GNN = auto()


@dataclass(frozen=True)
class AdapterField:
    """One routed observation field.

    Attributes:
        key: Field name, unique within one transform output.
        encoder: Encoder branch that consumes this field live. ``None`` means
            the field is not a live encoder input (buffer-only, e.g. compact
            replay image).
        buffer: Whether this field is stored in the replay buffer.
        value: The field payload for this step.
    """

    key: str
    encoder: Encoder 
    buffer: bool
    value: jnp.ndarray


AdapterOutput = list[AdapterField]


class FrameAdapterFn(Protocol):
    """Adapter needing only the raw env frame."""

    def __call__(self, frame: ObservationFrame) -> AdapterOutput: ...


class FeatureAdapterFn(Protocol):
    """Adapter needing the frame plus frozen VGGT features."""

    def __call__(
        self,
        frame: ObservationFrame,
        features: VGGTExtractOutput,
    ) -> AdapterOutput: ...


AdapterFn = FrameAdapterFn | FeatureAdapterFn


def ignore_features(fn: FrameAdapterFn) -> FeatureAdapterFn:
    """Lift a frame-only adapter to the feature signature (features unused).

    Lets the run loop deal with a single call shape when VGGT runs anyway;
    the branch on adapter kind happens once at wiring time, not per step.
    """
    return lambda frame, features: fn(frame)


def transition_from_fields(
    frame: ObservationFrame, fields: AdapterOutput
) -> ReplayTransition:
    """Build one replay transition from the routed adapter fields.

    ``buffer=True`` fields become the per-step replay observation; the single
    ``buffer=False`` field (if any) rides along as the live ``global_feature``
    (the buffer keeps only the latest value). Encoder routing travels as
    opaque int ids (``Encoder.value``) so the buffer never imports this
    module; recover it with :func:`routing_from_batch`.

    Raises:
        ValueError: If more than one field is routed ``buffer=False`` — the
            buffer has a single global-feature slot.
    """
    obs = {f.key: np.asarray(f.value) for f in fields if f.buffer}
    live = [f for f in fields if not f.buffer]
    if len(live) > 1:
        raise ValueError(
            f"only one live (buffer=False) field supported, got "
            f"{[f.key for f in live]}"
        )
    encoders = {f.key: f.encoder.value for f in fields}
    return ReplayTransition.from_frame(
        obs,
        frame,
        encoders=encoders,
        global_feature=live[0].value if live else None,
    )


def routing_from_batch(batch: ReplayBatch) -> dict[str, Encoder]:
    """Recover the ``key -> Encoder`` routing from a sampled batch."""
    return {key: Encoder(value) for key, value in (batch.encoders or {}).items()}
