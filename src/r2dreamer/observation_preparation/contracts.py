"""Observation Preparation contracts.

Observation Preparation turns raw environment observations into the replay-buffer
observation and the one-step agent observation for a chosen input mode. The
contract records the observation forms that keep replay storage, agent acting,
and the Encoder Module input aligned.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

import flax.linen as nn
import numpy as np

from src.configs.config import ObservationRunConfig

ObservationValue = np.ndarray | dict[str, np.ndarray]
ContractSnapshot = dict[str, Any]


def module_class_path(cls: type) -> str:
    """Return a stable import path for durable contract snapshots."""
    return f"{cls.__module__}.{cls.__qualname__}"


def _import_class(path: str) -> type:
    module_name, _, class_name = path.rpartition(".")
    if not module_name or not class_name:
        raise ValueError(f"invalid class path {path!r}")
    module = import_module(module_name)
    resolved: Any = module
    for part in class_name.split("."):
        resolved = getattr(resolved, part)
    if not isinstance(resolved, type):
        raise TypeError(f"{path!r} did not resolve to a class")
    return resolved


def encoder_module_kwargs_from_config(
    config: Any,
    encoder_module_cls: type | None = None,
) -> dict[str, Any]:
    """Resolve Encoder Module constructor kwargs from an effective config.

    Thin delegating shim: the config->kwargs formula lives on the launcher
    ``Encoder`` selection as ``module_kwargs_from_config``, co-located with
    its ``module_cls`` so the constructor signature and the resolved kwargs
    cannot desync (the structural fix for the Cause A drift). This function
    dispatches by ``config.encoder_type`` to the registered selection and
    returns its kwargs.

    ``encoder_module_cls`` is accepted for backwards compatibility — the
    launcher path and the unit tests still pass the resolved module class —
    and ignored; the launcher selection is the single source of truth.

    Args:
      config: Effective agent config (``R2DreamerConfig`` or equivalent)
        carrying ``encoder_type`` and the encoder knob values.
      encoder_module_cls: Deprecated, ignored. Retained so existing call sites
        (``train.py``, ``checkpointing.py``, ``module_factory.py``) need no
        change.

    Returns:
      Constructor kwargs for the selected Encoder Module.

    Raises:
      ValueError: If ``config.encoder_type`` is not a registered encoder.
    """
    from src.r2dreamer.launch.registries import encoder_registry

    encoder_type = getattr(config, "encoder_type", None)
    launcher_cls = encoder_registry.get(encoder_type)
    if launcher_cls is None:
        raise ValueError(
            f"no launcher Encoder registered for encoder_type {encoder_type!r}"
        )
    return launcher_cls.module_kwargs_from_config(config)


def normalize_encoder_module_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Recover tuple-valued constructor kwargs after JSON round-trips."""
    normalized = dict(kwargs)
    for key in ("mults", "cnn_mults", "conv_mults"):
        if key in normalized:
            normalized[key] = tuple(normalized[key])
    return normalized


@dataclass(frozen=True)
class ObservationField:
    """Metadata for one observation field."""

    shape: tuple[int, ...]
    dtype: str
    normalize_on_sample: bool = False

    def to_snapshot(self) -> ContractSnapshot:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "normalize_on_sample": self.normalize_on_sample,
        }

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> "ObservationField":
        return cls(
            shape=tuple(int(dim) for dim in snapshot["shape"]),
            dtype=str(snapshot["dtype"]),
            normalize_on_sample=bool(snapshot.get("normalize_on_sample", False)),
        )


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

    def to_snapshot(self) -> ContractSnapshot:
        if isinstance(self.fields, Mapping):
            return {
                "kind": "structured",
                "fields": {
                    key: field.to_snapshot() for key, field in self.fields.items()
                },
            }
        return {"kind": "single", "field": self.fields.to_snapshot()}

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Mapping[str, Any],
    ) -> "ObservationFormContract":
        kind = snapshot["kind"]
        if kind == "single":
            return cls(ObservationField.from_snapshot(snapshot["field"]))
        if kind == "structured":
            return cls(
                {
                    key: ObservationField.from_snapshot(field_snapshot)
                    for key, field_snapshot in snapshot["fields"].items()
                }
            )
        raise ValueError(f"unknown observation form snapshot kind {kind!r}")


def replay_observation_form(
    config: ObservationRunConfig,
) -> ObservationFormContract:
    """Build a replay observation contract from run observation config."""
    shapes = config.replay_field_shapes()
    dtypes = config.replay_field_dtypes()
    normalize = config.replay_field_normalize()
    fields = {
        name: ObservationField(shapes[name], dtypes[name], normalize[name])
        for name in shapes
    }
    if len(fields) == 1:
        return ObservationFormContract(next(iter(fields.values())))
    return ObservationFormContract(fields)


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
    encoder_module_kwargs: Mapping[str, Any] = field(default_factory=dict)
    design_notes: str = ""

    def to_snapshot(self) -> ContractSnapshot:
        return {
            "version": 1,
            "observation_preparation_type": self.observation_preparation_type,
            "encoder_type": self.encoder_type,
            "env_render_resolution": self.env_render_resolution,
            "encoder_module": module_class_path(self.encoder_module_cls),
            "env_observation": self.env_observation.to_snapshot(),
            "replay_observation": self.replay_observation.to_snapshot(),
            "agent_observation": self.agent_observation.to_snapshot(),
            "encoder_input": self.encoder_input.to_snapshot(),
            "decoder_target": (
                None
                if self.decoder_target is None
                else self.decoder_target.to_snapshot()
            ),
            "agent_overrides": dict(self.agent_overrides),
            "encoder_module_kwargs": dict(self.encoder_module_kwargs),
            "design_notes": self.design_notes,
        }

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Mapping[str, Any],
    ) -> "EncoderInputContract":
        if int(snapshot.get("version", 1)) != 1:
            raise ValueError(
                f"unsupported Encoder Input Contract snapshot version "
                f"{snapshot.get('version')!r}"
            )
        decoder_target = snapshot.get("decoder_target")
        return cls(
            observation_preparation_type=str(snapshot["observation_preparation_type"]),
            encoder_type=str(snapshot["encoder_type"]),
            env_render_resolution=int(snapshot["env_render_resolution"]),
            encoder_module_cls=_import_class(str(snapshot["encoder_module"])),
            env_observation=ObservationFormContract.from_snapshot(
                snapshot["env_observation"]
            ),
            replay_observation=ObservationFormContract.from_snapshot(
                snapshot["replay_observation"]
            ),
            agent_observation=ObservationFormContract.from_snapshot(
                snapshot["agent_observation"]
            ),
            encoder_input=ObservationFormContract.from_snapshot(
                snapshot["encoder_input"]
            ),
            decoder_target=(
                None
                if decoder_target is None
                else ObservationFormContract.from_snapshot(decoder_target)
            ),
            agent_overrides=dict(snapshot.get("agent_overrides", {})),
            encoder_module_kwargs=normalize_encoder_module_kwargs(
                snapshot.get("encoder_module_kwargs", {})
            ),
            design_notes=str(snapshot.get("design_notes", "")),
        )


def recover_encoder_input_contract(
    snapshot: Mapping[str, Any],
) -> EncoderInputContract:
    """Recover a runtime Encoder Input Contract from its durable snapshot."""
    return EncoderInputContract.from_snapshot(snapshot)


@dataclass(frozen=True)
class PreparedObservation:
    """Per-step prepared observation for replay storage and immediate acting."""

    replay_obs: ObservationValue
    encoder_obs: Any
    is_first: bool
