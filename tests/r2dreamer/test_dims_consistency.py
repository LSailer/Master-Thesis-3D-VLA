"""Guard: VGGT/hybrid feature dimensions must agree across the layers that each
declare them.

The VGGT WP/CP feature size (37*37*3 + 9 = 4116) is referenced in three different
architectural layers that — for good coupling reasons — do not import each other:

  * ``adapters.vggt_adapter.VGGT_FEATURE_DIM``     (buffer/extractor layer; canonical)
  * ``world_model.encoders.HYBRID_VGGT_DIM``       (encoder slice size)
  * ``config.R2DreamerConfig.vggt_feature_dim``    (config default)

Importing one into the others would drag the heavy VGGT/Flax stack into the
lightweight config/encoder modules, so instead this test pins them equal and fails
fast if a grid-size ablation updates one but not the others. Same idea for the
hybrid buffer layout = RGB branch + VGGT branch.
"""

from __future__ import annotations

import dataclasses

from src.r2dreamer.adapters.hybrid_adapter import HYBRID_FEATURE_DIM
from src.r2dreamer.adapters.vggt_adapter import VGGT_FEATURE_DIM
from src.r2dreamer.config import R2DreamerConfig
from src.r2dreamer.world_model.encoders import HYBRID_RGB_DIM, HYBRID_VGGT_DIM


def _field_default(cls, name: str):
    return next(f for f in dataclasses.fields(cls) if f.name == name).default


def test_encoder_vggt_slice_matches_adapter_feature_dim():
    assert HYBRID_VGGT_DIM == VGGT_FEATURE_DIM


def test_config_default_matches_adapter_feature_dim():
    assert _field_default(R2DreamerConfig, "vggt_feature_dim") == VGGT_FEATURE_DIM


def test_hybrid_feature_dim_is_sum_of_branches():
    assert HYBRID_FEATURE_DIM == HYBRID_RGB_DIM + HYBRID_VGGT_DIM
