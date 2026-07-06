"""Tests for deterministic static RGB point-cloud house context encoding."""
# pylint: disable=missing-function-docstring

import numpy as np
import pytest

from src.r2dreamer.encoders.constants import HOUSE_CONTEXT_DIM
from src.r2dreamer.observation_preparation.static_house_context import (
    PlyFormatError,
    encode_static_house_context,
    load_ascii_ply_xyzrgb,
)


def test_static_house_context_is_1024_float16_and_order_invariant():
    points = np.array(
        [
            [0.0, 0.0, 0.0, 255.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 255.0, 0.0],
            [0.2, 0.4, 0.1, 0.0, 0.0, 255.0],
            [0.8, 0.1, 0.5, 128.0, 64.0, 32.0],
        ],
        dtype=np.float32,
    )

    context = encode_static_house_context(points)
    shuffled = encode_static_house_context(points[::-1])

    assert context.shape == (HOUSE_CONTEXT_DIM,)
    assert context.dtype == np.float16
    assert np.isfinite(context).all()
    np.testing.assert_allclose(context, shuffled)


def test_static_house_context_preserves_rgb_channels_and_filters_nonfinite_points():
    points = np.array(
        [
            [0.0, 0.0, 0.0, 255.0, 0.0, 0.0],
            [np.nan, 0.1, 0.1, 0.0, 255.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0, 255.0],
        ],
        dtype=np.float32,
    )

    context = encode_static_house_context(points)

    assert context[0] == pytest.approx(1.0)
    assert context[1] == pytest.approx(1.0)
    assert context[2] == pytest.approx(0.0)
    assert context[3] == pytest.approx(0.0)
    assert float(np.max(context)) <= 1.0


def test_load_ascii_ply_xyzrgb_reads_required_vertex_fields(tmp_path):
    ply_path = tmp_path / "house.ply"
    ply_path.write_text(
        "\n".join(
            [
                "ply",
                "format ascii 1.0",
                "element vertex 2",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "0.0 0.0 0.0 255 0 0",
                "1.0 1.0 1.0 0 0 255",
            ]
        ),
        encoding="ascii",
    )

    points = load_ascii_ply_xyzrgb(ply_path)
    context = encode_static_house_context(load_ascii_ply_xyzrgb(ply_path))

    assert points.shape == (2, 6)
    np.testing.assert_allclose(points[0], [0.0, 0.0, 0.0, 255.0, 0.0, 0.0])
    assert context.shape == (HOUSE_CONTEXT_DIM,)


def test_load_ascii_ply_xyzrgb_rejects_binary_ply(tmp_path):
    ply_path = tmp_path / "binary.ply"
    ply_path.write_text(
        "\n".join(
            [
                "ply",
                "format binary_little_endian 1.0",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
            ]
        ),
        encoding="ascii",
    )

    with pytest.raises(PlyFormatError, match="ascii"):
        load_ascii_ply_xyzrgb(ply_path)
