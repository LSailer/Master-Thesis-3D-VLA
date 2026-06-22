"""Encoder Module construction from Observation Preparation contracts.

This is the agent-facing edge of the Observation Preparation boundary: the
agent asks for a resolved Encoder Input Contract and instantiates the Dreamer
Encoder Module from the class + kwargs recorded there. Legacy direct
``R2DreamerConfig`` construction without a durable contract is still supported
by deriving a minimal contract from the effective config.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp

from src.r2dreamer.config import R2DreamerConfig
from src.r2dreamer.obs_batch import HYBRID_IMAGE_KEY
from src.r2dreamer.observation_preparation.cnn import CNNObservationPreparation
from src.r2dreamer.observation_preparation.contracts import (
    EncoderInputContract,
    ObservationField,
    ObservationFormContract,
    encoder_module_kwargs_from_config,
    normalize_encoder_module_kwargs,
    recover_encoder_input_contract,
)
from src.r2dreamer.observation_preparation.vggt import VGGT_DREAMER_SPECS
from src.r2dreamer.world_model.encoders import HYBRID_RGB_DIM
from src.r2dreamer.world_model.rssm import R2RSSM


_DECODER_RGB_ENCODERS = frozenset(
    {
        "cnn",
        "hybrid",
        "vggt_house_context",
        "vggt_house_full_tokens_nogate",
        "vggt_house_global_tokens_nogate",
    }
)


def make_rssm_module(cfg: R2DreamerConfig) -> R2RSSM:
    """Build the RSSM module from config."""
    return R2RSSM(
        deter_size=cfg.deter_size,
        stoch_classes=cfg.stoch_classes,
        stoch_discrete=cfg.stoch_discrete,
        num_actions=cfg.num_actions,
        hidden=cfg.hidden_size,
        blocks=cfg.blocks,
        dyn_layers=cfg.dyn_layers,
        obs_layers=cfg.obs_layers,
        img_layers=cfg.img_layers,
        unimix_ratio=cfg.unimix_ratio,
    )


def _encoder_module_cls_from_config(cfg: R2DreamerConfig):
    if cfg.encoder_module_cls is not None:
        return cfg.encoder_module_cls
    if cfg.encoder_type == "cnn":
        return CNNObservationPreparation().contract.encoder_module_cls
    spec = VGGT_DREAMER_SPECS.get(cfg.encoder_type)
    if spec is not None:
        return spec.module_cls
    raise ValueError(f"unknown encoder_type {cfg.encoder_type!r}")


def _field(shape: tuple[int, ...], dtype: str = "float32") -> ObservationField:
    return ObservationField(tuple(int(dim) for dim in shape), dtype)


def _form_from_shape(
    shape: tuple[int, ...] | Mapping[str, tuple[int, ...]],
    *,
    dtype: str = "float32",
) -> ObservationFormContract:
    if isinstance(shape, Mapping):
        return ObservationFormContract(
            {key: _field(tuple(value), dtype) for key, value in shape.items()}
        )
    return ObservationFormContract(_field(tuple(shape), dtype))


def _legacy_env_observation(cfg: R2DreamerConfig) -> ObservationFormContract:
    image_shape = (3, 64, 64)
    if cfg.encoder_type.startswith("vggt") and cfg.encoder_type not in _DECODER_RGB_ENCODERS:
        image_shape = (3, 518, 518)
    return ObservationFormContract(
        {
            HYBRID_IMAGE_KEY: ObservationField(image_shape, "uint8"),
            "is_first": ObservationField((), "bool"),
        }
    )


def _legacy_agent_observation(cfg: R2DreamerConfig) -> ObservationFormContract:
    if cfg.encoder_type in (
        "vggt",
        "vggt_wp_cp_64",
        "vggt_aggregator_mlp",
        "vggt_agg_token_transformer",
        "vggt_wp_dense_cnn",
    ):
        return ObservationFormContract(
            {
                "features": _field(tuple(cfg.obs_shape)),  # type: ignore[arg-type]
                "is_first": ObservationField((), "bool"),
            }
        )
    if isinstance(cfg.obs_shape, Mapping):
        fields = {key: _field(tuple(value)) for key, value in cfg.obs_shape.items()}
        fields["is_first"] = ObservationField((), "bool")
        return ObservationFormContract(fields)
    return ObservationFormContract(
        {
            HYBRID_IMAGE_KEY: _field(tuple(cfg.obs_shape)),
            "is_first": ObservationField((), "bool"),
        }
    )


def _legacy_decoder_target(cfg: R2DreamerConfig) -> ObservationFormContract | None:
    if cfg.encoder_type not in _DECODER_RGB_ENCODERS:
        return None
    return ObservationFormContract(ObservationField((3, 64, 64), "float32"))


def legacy_encoder_input_contract_from_config(
    cfg: R2DreamerConfig,
) -> EncoderInputContract:
    """Derive a minimal contract for legacy direct config construction.

    Launch/evaluation code should pass a serialized contract snapshot. This
    fallback keeps older tests, profiling scripts, and checkpoint loaders alive
    while centralizing the remaining string compatibility outside ``agent.py``.
    """
    if cfg.encoder_type == "cnn":
        contract = CNNObservationPreparation().contract
        return EncoderInputContract(
            observation_preparation_type=contract.observation_preparation_type,
            encoder_type=contract.encoder_type,
            env_render_resolution=contract.env_render_resolution,
            encoder_module_cls=contract.encoder_module_cls,
            env_observation=contract.env_observation,
            replay_observation=contract.replay_observation,
            agent_observation=contract.agent_observation,
            encoder_input=_form_from_shape(cfg.obs_shape),
            decoder_target=contract.decoder_target,
            agent_overrides=contract.agent_overrides,
            encoder_module_kwargs=encoder_module_kwargs_from_config(
                cfg,
                contract.encoder_module_cls,
            ),
            design_notes=contract.design_notes,
        )

    cls = _encoder_module_cls_from_config(cfg)
    return EncoderInputContract(
        observation_preparation_type=cfg.encoder_type,
        encoder_type=cfg.encoder_type,
        env_render_resolution=518 if cfg.encoder_type.startswith("vggt") else 64,
        encoder_module_cls=cls,
        env_observation=_legacy_env_observation(cfg),
        replay_observation=_form_from_shape(cfg.obs_shape),
        agent_observation=_legacy_agent_observation(cfg),
        encoder_input=_form_from_shape(cfg.obs_shape),
        decoder_target=_legacy_decoder_target(cfg),
        agent_overrides={},
        encoder_module_kwargs=encoder_module_kwargs_from_config(cfg, cls),
        design_notes=getattr(cfg, "design_notes", ""),
    )


def resolve_encoder_input_contract(cfg: R2DreamerConfig) -> EncoderInputContract:
    """Return the effective Encoder Input Contract for an agent config."""
    snapshot = getattr(cfg, "encoder_input_contract", None)
    if snapshot is not None:
        return recover_encoder_input_contract(snapshot)
    return legacy_encoder_input_contract_from_config(cfg)


def _validate_encoder_module_config(
    cfg: R2DreamerConfig,
    contract: EncoderInputContract,
) -> None:
    class_name = contract.encoder_module_cls.__name__
    if class_name in {"ConvEncoder", "WP64CNNCPMLPEncoder"} and cfg.vggt_mlp_layers != 1:
        raise ValueError(
            f"vggt_mlp_layers={cfg.vggt_mlp_layers} has no effect on "
            f"{class_name} (a conv encoder, no MLP blocks). Only the 'vggt' and "
            f"'vggt_aggregator_mlp' encoders consume vggt_mlp_layers; leave it at 1 "
            f"for cnn / vggt_wp_dense_cnn."
        )
    if class_name != "HybridEncoder":
        return
    expected_shape = (HYBRID_RGB_DIM + cfg.vggt_feature_dim,)
    if not isinstance(cfg.obs_shape, tuple):
        raise ValueError(f"hybrid expects flat obs_shape, got {cfg.obs_shape}")
    if not (
        cfg.obs_shape == expected_shape
        and cfg.obs_shape[0] - cfg.vggt_feature_dim == HYBRID_RGB_DIM
    ):
        raise ValueError(
            "hybrid obs_shape/split mismatch: expected "
            f"{expected_shape} with vggt_feature_dim={cfg.vggt_feature_dim}, "
            f"got obs_shape={cfg.obs_shape}, "
            f"vggt_feature_dim={cfg.vggt_feature_dim}"
        )


def make_encoder_module(cfg: R2DreamerConfig):
    """Instantiate the Dreamer Encoder Module from the resolved contract."""
    contract = resolve_encoder_input_contract(cfg)
    _validate_encoder_module_config(cfg, contract)
    kwargs = normalize_encoder_module_kwargs(contract.encoder_module_kwargs)
    if not kwargs:
        kwargs = encoder_module_kwargs_from_config(cfg, contract.encoder_module_cls)
    return contract.encoder_module_cls(**kwargs)


def _dummy_field(field: ObservationField) -> jnp.ndarray:
    dtype = jnp.uint8 if field.dtype == "uint8" else jnp.float32
    return jnp.zeros((1, *field.shape), dtype=dtype)


def dummy_encoder_input(cfg: R2DreamerConfig):
    """Create a batch-sized dummy input matching the Encoder Module contract."""
    contract = resolve_encoder_input_contract(cfg)
    fields = contract.encoder_input.fields
    if isinstance(fields, Mapping):
        return {key: _dummy_field(field) for key, field in fields.items()}
    return _dummy_field(fields)


def require_decoder_target(cfg: R2DreamerConfig) -> None:
    """Raise when decoder=True is incompatible with the selected input mode."""
    contract = resolve_encoder_input_contract(cfg)
    if contract.decoder_target is not None:
        return
    raise ValueError(
        "decoder=True requires an Observation Preparation with an RGB decoder "
        "target, but "
        f"{contract.encoder_type!r} exposes no decoder_target."
    )
