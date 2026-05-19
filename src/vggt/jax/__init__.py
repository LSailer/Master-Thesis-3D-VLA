"""JAX/Flax reimplementation of StreamVGGT (InfiniteVGGT).

Steps 1-6 ship the core model (weight transfer, DINOv2 backbone,
aggregator, camera head, DPT point head, streaming KV-cache with dynamic
budget eviction). Step 7 adds ``JAXVGGTFeatureExtractor`` as a drop-in
replacement for the PyTorch extractor.
"""

from src.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor
from src.vggt.jax.weight_transfer import (
    load_checkpoint,
    load_pytorch_weights,
    V1_EXCLUDE_PREFIXES,
)

__all__ = [
    "JAXVGGTFeatureExtractor",
    "V1_EXCLUDE_PREFIXES",
    "load_checkpoint",
    "load_pytorch_weights",
]
