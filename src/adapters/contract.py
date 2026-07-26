"""Adapter contract: one call per frame returns a flat list of routed fields.

Each field says which encoder branch consumes it (``encoder``), whether it is
stored in replay (``buffer``), and whether the decoder probe reconstructs it
(``decoder_target``). ``is_first`` is NOT part of the output - the collector
passes it explicitly when appending to the buffer.

This module is the single place where observation routing is declared. No
encoder-type string, no parallel shape table: the adapter emits the value and
its routing together, and every consumer (replay buffer, composite encoder,
decoder probe) reads the routing off the fields.

Frozen extraction lives inside the adapter: a variant that needs VGGT takes a
:class:`FeatureExtractor` at construction and calls it itself, so every adapter
has the same one-argument call shape and the collector never sees VGGT.
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
    """Trainable encoder branch that consumes a field live.

    Every member must have a case in
    :meth:`src.r2dreamer.encoders.routed_composite.RoutedCompositeEncoder._branch`,
    which resolves it to a module in ``src/r2dreamer/encoders/``; that module
    validates the field shape it was handed. A member no adapter routes to does
    not belong here.
    """

    CONV = auto()
    # Spatial convolution over a metric point map rather than an image: same
    # architecture, but symlog compression instead of RGB centering, because
    # world coordinates are unbounded.
    CONV_POINTS = auto()
    MLP = auto()
    POINTNET = auto()
    GNN = auto()
    TRANSFORMER = auto()


@dataclass(frozen=True)
class AdapterField:
    """One routed observation field.

    Attributes:
        key: Field name, unique within one adapter output.
        encoder: Encoder branch that consumes this field live.
        buffer: Whether this field is stored in the replay buffer. Exactly one
            field per adapter may be ``False``: that value is not replayed but
            rides along as the live ``global_feature`` (latest value only).
        value: The field payload for this step.
        decoder_target: Whether the debug decoder probe reconstructs this
            field. At most one field per adapter, and it must be replayed
            (``buffer=True``) because the probe reads its target from the
            sampled batch.
    """

    key: str
    encoder: Encoder
    buffer: bool
    value: jnp.ndarray
    decoder_target: bool = False


AdapterOutput = list[AdapterField]


class AdapterFn(Protocol):
    """Turns one env frame into routed fields.

    The only adapter shape. Variants that need frozen features hold their
    extractor and call it inside this method, so callers never branch on the
    kind of adapter they hold.
    """

    def __call__(self, frame: ObservationFrame) -> AdapterOutput: ...


class FeatureExtractor(Protocol):
    """The slice of the frozen VGGT extractor an adapter needs.

    Handed the whole frame, not just the image: the extractor owns its cache
    lifetime and reads ``is_first``/``scene_id`` off the frame to apply the reset
    policy it was constructed with (``ResetMode.FULL`` wipes per episode,
    ``PERSIST_SCENE`` saves and restores per scene). Cache lifetime therefore
    never leaks into adapters or the collector, and the reset cannot end up
    ordered after the first extract of an episode.
    """

    def extract(self, source: ObservationFrame) -> VGGTExtractOutput: ...


def decoder_target_key(fields: AdapterOutput) -> str:
    """Return the key of the single field the decoder probe reconstructs.

    Args:
        fields: Adapter output for one representative frame.

    Returns:
        The ``decoder_target`` field's key.

    Raises:
        ValueError: If not exactly one field is flagged, or if the flagged
            field is not replayed (the probe reads targets from replay).
    """
    targets = [f for f in fields if f.decoder_target]
    if len(targets) != 1:
        raise ValueError(
            "decoder=True needs exactly one decoder_target field, got "
            f"{[f.key for f in targets]}"
        )
    target = targets[0]
    if not target.buffer:
        raise ValueError(
            f"decoder_target field {target.key!r} must be stored in replay "
            "(buffer=True) - the probe reads its target from the sampled batch"
        )
    return target.key


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


def encoder_obs_from_fields(fields: AdapterOutput) -> dict[str, jnp.ndarray]:
    """Return the live encoder observation for one step (no leading dims).

    Both replayed and live fields are included: acting needs the same obs dict
    the encoder saw at init, just for a single env step. The agent adds the
    batch dim, skipping live fields (which stay one global event).
    """
    return {f.key: f.value for f in fields}


def routing_from_batch(batch: ReplayBatch) -> dict[str, Encoder]:
    """Recover the ``key -> Encoder`` routing from a sampled batch.

    Not dead code, and not how the encoder is built: the model is deliberately
    composed from the adapter's live fields in ``R2DreamerAgent.__init__``, not
    from ``batch.encoders``. The routing rides through replay so the buffer can
    refuse to mix rows whose routing changed mid-run (see the ``ValueError`` in
    :meth:`src.buffer.replay_buffer.ReplayBuffer.add`) - a failure mode an
    adapter-declared composition makes reachable - and so a stored batch stays
    interpretable on its own.
    """
    return {key: Encoder(value) for key, value in (batch.encoders or {}).items()}
