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
from src.r2dreamer.encoders.constants import AGG_REGISTER_TOKENS
from src.r2dreamer.observation_keys import (
    FULL_TOKENS_KEY,
    GLOBAL_TOKENS_KEY,
    HYBRID_IMAGE_KEY,
)

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


def _tuple_value(config: Any, name: str) -> tuple[int, ...]:
    return tuple(getattr(config, name))


# Encoder Module classes with a dedicated kwargs branch below. Matched by
# name (not issubclass) so this module stays import-free of the encoder
# packages, which import contract machinery themselves.
_HANDLED_ENCODER_CLASS_NAMES = frozenset(
    {
        "ConvEncoder",
        "WP64CNNCPMLPEncoder",
        "HybridEncoder",
        "HousePointsCameraEncoder",
        "HybridHousePointsCameraEncoder",
        "HouseGlobalEmbeddingEncoder",
        "TokenTransformerEncoder",
    }
)


def _kwargs_dispatch_name(encoder_module_cls: type) -> str:
    """Return the nearest handled class name along the MRO.

    Subclasses (e.g. the GNN variants of ``HousePointsCameraEncoder``)
    inherit their parent's constructor kwargs, so dispatch walks the MRO
    and picks the first ancestor with a dedicated kwargs branch — the
    name-matching equivalent of an ``issubclass`` chain ordered
    most-derived-first.

    Args:
      encoder_module_cls: The Encoder Module class to resolve kwargs for.

    Returns:
      The matched ancestor's class name, or the class's own name when no
      ancestor has a dedicated branch (falls through to the MLP tail).
    """
    for base in encoder_module_cls.__mro__:
        if base.__name__ in _HANDLED_ENCODER_CLASS_NAMES:
            return base.__name__
    return encoder_module_cls.__name__


def encoder_module_kwargs_from_config(
    config: Any,
    encoder_module_cls: type,
) -> dict[str, Any]:
    """Resolve Encoder Module constructor kwargs from an effective config.

    Dispatch is subclass-aware: a class without its own branch inherits the
    kwargs of its nearest handled ancestor (see ``_kwargs_dispatch_name``).

    Args:
      config: Effective agent config (``R2DreamerConfig`` or equivalent).
      encoder_module_cls: The Encoder Module class to be constructed.

    Returns:
      Constructor kwargs for ``encoder_module_cls``.
    """
    class_name = _kwargs_dispatch_name(encoder_module_cls)
    if class_name == "ConvEncoder":
        kwargs = {
            "depth": int(config.encoder_depth),
            "kernel_size": int(config.encoder_kernel),
            "mults": _tuple_value(config, "encoder_mults"),
        }
        if getattr(config, "encoder_type", None) == "vggt_wp_dense_cnn":
            kwargs.update(
                {
                    "input_kind": "world_points",
                    "embed_dim": int(config.vggt_embed_dim),
                }
            )
        return kwargs
    if class_name == "WP64CNNCPMLPEncoder":
        return {
            "embed_dim": int(config.vggt_embed_dim),
            "conv_depth": int(config.encoder_depth),
            "conv_kernel": int(config.encoder_kernel),
            "conv_mults": _tuple_value(config, "encoder_mults"),
            "cp_hidden": int(config.mlp_vggt_hidden),
            "cp_layers": int(config.mlp_vggt_layers),
        }
    if class_name == "HybridEncoder":
        return {
            "cnn_depth": int(config.encoder_depth),
            "cnn_kernel": int(config.encoder_kernel),
            "cnn_mults": _tuple_value(config, "encoder_mults"),
            "vggt_embed_dim": int(config.vggt_embed_dim),
            "mlp_hidden": int(config.mlp_vggt_hidden),
            "mlp_layers": int(config.mlp_vggt_layers),
            "vggt_dim": int(config.vggt_feature_dim),
        }
    if class_name == "HousePointsCameraEncoder":
        return {
            "embed_dim": int(config.vggt_embed_dim),
            "camera_hidden": int(config.mlp_vggt_hidden),
            "camera_layers": int(config.mlp_vggt_layers),
            "point_hidden": int(config.mlp_vggt_hidden),
            "point_layers": int(config.mlp_vggt_layers),
        }
    if class_name == "HybridHousePointsCameraEncoder":
        return {
            "embed_dim": int(config.vggt_embed_dim),
            "camera_hidden": int(config.mlp_vggt_hidden),
            "camera_layers": int(config.mlp_vggt_layers),
            "point_hidden": int(config.mlp_vggt_hidden),
            "point_layers": int(config.mlp_vggt_layers),
            "cnn_depth": int(config.encoder_depth),
            "cnn_kernel": int(config.encoder_kernel),
            "cnn_mults": _tuple_value(config, "encoder_mults"),
        }
    if class_name == "HouseGlobalEmbeddingEncoder":
        # token_dim (1024) and num_patch_tokens (1369) use the module defaults
        # — they are fixed by the VGGT global-half token layout.
        return {
            "embed_dim": int(config.vggt_embed_dim),
            "token_dim": int(config.vggt_token_dim),
            "num_patch_tokens": int(config.vggt_token_count)
            - (1 + AGG_REGISTER_TOKENS),
            "reducer_hidden": int(config.mlp_vggt_hidden),
            "reducer_layers": int(config.mlp_vggt_layers),
            "camera_hidden": int(config.mlp_vggt_hidden),
            "camera_layers": int(config.mlp_vggt_layers),
            "rgb_branch": getattr(config, "vggt_house_global_rgb_branch", False),
        }
    if class_name == "TokenTransformerEncoder":
        common = {
            "embed_dim": int(config.vggt_embed_dim),
            "token_dim": int(config.vggt_token_dim),
            "num_tokens": int(config.vggt_token_count),
            "layers": int(config.vggt_token_transformer_layers),
            "heads": int(config.vggt_token_transformer_heads),
            "mlp_ratio": int(config.vggt_token_transformer_mlp_ratio),
            "dropout": float(config.vggt_token_transformer_dropout),
        }
        if getattr(config, "encoder_type", None) == "vggt_agg_token_transformer":
            return {
                **common,
                "model_dim": int(config.vggt_token_projection_dim),
                "readout": "camera_register_patch",
                "norm_kind": "rms",
                "activation": "silu",
                "keep_register_tokens": bool(config.vggt_keep_register_tokens),
            }
        token_key = FULL_TOKENS_KEY
        singleton_tokens = False
        if getattr(config, "encoder_type", None) == "vggt_house_global_tokens_nogate":
            token_key = GLOBAL_TOKENS_KEY
            singleton_tokens = True
        return {
            **common,
            "model_dim": None,
            "readout": "mean",
            "norm_kind": "layer",
            "activation": "gelu",
            "token_key": token_key,
            "image_key": HYBRID_IMAGE_KEY,
            "singleton_tokens": singleton_tokens,
            "cnn_depth": int(config.encoder_depth),
            "cnn_kernel": int(config.encoder_kernel),
            "cnn_mults": _tuple_value(config, "encoder_mults"),
        }
    return {
        "embed_dim": int(config.vggt_embed_dim),
        "hidden": int(config.vggt_embed_dim),
        "num_layers": int(config.vggt_mlp_layers),
    }


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
