"""CPU tests for consumed-layer token-half exposure and the point-head re-entry.

Covers the shape/validation contracts of
``JAXVGGTFeatureExtractor.consumed_layer_halves`` and
``point_head_from_tokens`` (openspec change
``global-token-reconstruction-ablation``, tasks 2.x / 3.1-3.2) on shell
instances without GPU or checkpoint weights. The bit-for-bit reproduction
check (task 3.3) requires the real point head and runs inside
``src/prototyp/global_token_reconstruction/run_three_arm_ablation.py``.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from src.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor
from src.vggt.jax.heads.dpt_head import DPTHead

_N_TOKENS = 5 + 37**2  # camera + register tokens, then 1369 patch tokens
_N_PATCH = 37**2
_WIDTH = 2048
_DEPTH = 24
_CONSUMED = (4, 11, 17, 23)


def _shell_extractor() -> JAXVGGTFeatureExtractor:
    """Build a minimal extractor shell without running __init__ (no weights)."""
    extractor = JAXVGGTFeatureExtractor.__new__(JAXVGGTFeatureExtractor)
    extractor._point_head = DPTHead()
    extractor._agg_depth = _DEPTH
    extractor._last_out_list = None
    extractor._last_patch_start_idx = None
    return extractor


def _with_fake_out_list(extractor: JAXVGGTFeatureExtractor) -> None:
    """Attach a deterministic fake 24-layer aggregator output list."""
    rng = np.random.default_rng(0)
    extractor._last_out_list = [
        jnp.asarray(
            rng.normal(size=(1, 1, _N_TOKENS, _WIDTH)).astype(np.float32)
        )
        for _ in range(_DEPTH)
    ]
    extractor._last_patch_start_idx = jnp.asarray(5, dtype=jnp.int32)


def test_consumed_layer_halves_shapes_and_split():
    extractor = _shell_extractor()
    _with_fake_out_list(extractor)

    halves = extractor.consumed_layer_halves()

    assert set(halves) == set(_CONSUMED)
    for layer_idx, (frame_half, global_half) in halves.items():
        assert frame_half.shape == (_N_PATCH, _WIDTH // 2)
        assert global_half.shape == (_N_PATCH, _WIDTH // 2)
        patch_tokens = extractor._last_out_list[layer_idx][0, 0, 5:, :]
        np.testing.assert_array_equal(
            np.asarray(frame_half), np.asarray(patch_tokens[:, : _WIDTH // 2])
        )
        np.testing.assert_array_equal(
            np.asarray(global_half), np.asarray(patch_tokens[:, _WIDTH // 2 :])
        )


def test_consumed_layer_halves_raises_before_extract():
    extractor = _shell_extractor()
    with pytest.raises(RuntimeError, match="before any extract"):
        extractor.consumed_layer_halves()


def test_reentry_rejects_missing_or_extra_layers():
    extractor = _shell_extractor()
    good = jnp.zeros((_N_PATCH, _WIDTH), dtype=jnp.float32)
    with pytest.raises(ValueError, match="exactly the consumed layers"):
        extractor.point_head_from_tokens({4: good, 11: good, 17: good})
    with pytest.raises(ValueError, match="exactly the consumed layers"):
        extractor.point_head_from_tokens(
            {4: good, 11: good, 17: good, 23: good, 0: good}
        )


@pytest.mark.parametrize(
    "bad_shape",
    [
        (_N_PATCH, _WIDTH // 2),  # a bare half, not the 2048-wide concat
        (1, _WIDTH),  # pooled scene vector broadcast into the patch slot
        (_N_PATCH + 5, _WIDTH),  # full token sequence incl. camera/register
    ],
)
def test_reentry_rejects_non_per_patch_2048_tokens(bad_shape):
    extractor = _shell_extractor()
    tokens = {
        idx: jnp.zeros((_N_PATCH, _WIDTH), dtype=jnp.float32) for idx in _CONSUMED
    }
    tokens[23] = jnp.zeros(bad_shape, dtype=jnp.float32)
    with pytest.raises(ValueError, match="expected"):
        extractor.point_head_from_tokens(tokens)
