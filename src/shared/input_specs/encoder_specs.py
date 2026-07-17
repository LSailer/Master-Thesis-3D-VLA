"""Encoder specs: compose readouts into a named observation/encoder contract.

An ``EncoderSpec`` is the unit registered in ``ENCODER_SPECS``. Each readout
owns preparation and its encoder branch; a hybrid is simply a spec with more
than one readout. Package-independent of ``r2dreamer`` / ``vggt``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from src.shared.input_specs.readout import Readout


@dataclass(frozen=True)
class EncoderSpec:
    """Named composition of observation readouts for one encoder configuration.

    Attributes:
        name: Registry key for this spec (e.g. ``\"cnn\"``, ``\"hybrid\"``).
        readouts: Ordered readouts taken from each ``ObservationFrame``. Each
            carries its own encoder branch; more than one readout means a
            fused (hybrid) encoder input.
        agent_overrides: Optional agent/config overrides applied when this
            spec is selected (e.g. embed dims, module kwargs).
    """

    name: str
    readouts: tuple[Readout, ...]
    agent_overrides: Mapping[str, Any] = field(default_factory=dict)



# # Your hybrid — replayed RGB for the CNN branch, live global tokens:
# "cnn_plus_house_global": EncoderSpec(
#     name="cnn_plus_house_global",
#     readouts=(
#         RGB(size=64),                                   # -> cnn (only option)
#         Tokens(half="global", encoder="transformer",    # mlp also legal
#                source=LIVE_LATEST),
#     ),
# ),