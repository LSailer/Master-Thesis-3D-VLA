"""JAX/Flax reimplementation of StreamVGGT (InfiniteVGGT).

Public entry points are exposed here as they land. Step 1 only ships
weight-transfer; Flax module shells land in subsequent steps.
"""

from modules.vggt.jax.weight_transfer import (
    load_checkpoint,
    load_pytorch_weights,
    V1_EXCLUDE_PREFIXES,
)

__all__ = ["load_checkpoint", "load_pytorch_weights", "V1_EXCLUDE_PREFIXES"]
