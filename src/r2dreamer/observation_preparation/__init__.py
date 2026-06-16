"""Observation Preparation public API."""

from src.r2dreamer.observation_preparation.contracts import (
    EncoderInputContract,
    ObservationField,
    ObservationFormContract,
    PreparedObservation,
)
from src.r2dreamer.observation_preparation.cnn import (
    CNN_IMAGE_SHAPE,
    CNNObservationPreparation,
)
from src.r2dreamer.observation_preparation.vggt import (
    HYBRID_FEATURE_DIM,
    HYBRID_IMAGE_SHAPE,
    VGGTFeatureKind,
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
    "ObservationField",
    "ObservationFormContract",
    "PreparedObservation",
    "VGGTFeatureKind",
    "VGGT_IMAGE_SHAPE",
    "build_hybrid_contract",
    "build_vggt_contract",
]
