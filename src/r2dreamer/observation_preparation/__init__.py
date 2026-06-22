"""Observation Preparation public API."""

from src.r2dreamer.observation_preparation.contracts import (
    EncoderInputContract,
    ObservationField,
    ObservationFormContract,
    PreparedObservation,
    encoder_module_kwargs_from_config,
    module_class_path,
    normalize_encoder_module_kwargs,
    recover_encoder_input_contract,
    replay_observation_form,
)
from src.r2dreamer.observation_preparation.cnn import (
    CNN_IMAGE_SHAPE,
    CNNObservationPreparation,
)
from src.r2dreamer.observation_preparation.vggt import (
    HYBRID_FEATURE_DIM,
    HYBRID_IMAGE_SHAPE,
    DreamerEncoderSpec,
    HeadReadout,
    StorageSpec,
    TokenReadout,
    VGGT_DREAMER_SPECS,
    VGGTFeatureKind,
    VGGTDreamerSpec,
    VGGT_IMAGE_SHAPE,
    build_hybrid_contract,
    build_vggt_contract,
)

__all__ = [
    "CNN_IMAGE_SHAPE",
    "CNNObservationPreparation",
    "EncoderInputContract",
    "HYBRID_FEATURE_DIM",
    "HYBRID_IMAGE_SHAPE",
    "DreamerEncoderSpec",
    "ObservationField",
    "ObservationFormContract",
    "PreparedObservation",
    "encoder_module_kwargs_from_config",
    "module_class_path",
    "normalize_encoder_module_kwargs",
    "recover_encoder_input_contract",
    "replay_observation_form",
    "HeadReadout",
    "StorageSpec",
    "TokenReadout",
    "VGGT_DREAMER_SPECS",
    "VGGTFeatureKind",
    "VGGTDreamerSpec",
    "VGGT_IMAGE_SHAPE",
    "build_hybrid_contract",
    "build_vggt_contract",
]
