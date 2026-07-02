"""Roundtrip and header-tolerance checks for ``src.prototype_helpers.ply_io``."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from src.prototype_helpers.ply_io import load_ply_xyzrgb, save_ply_xyzrgb

_BENCH_STYLE_PLY = """ply
format ascii 1.0
comment color legend: 2=new orange
element vertex 3
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property int point_id
property int status_id
property int added_step
end_header
0.5 -1.25 2.0 10 20 30 0 2 0
1.5 0.0 -0.5 40 50 60 1 2 0
-3.0 4.0 0.25 70 80 90 2 2 1
"""


def test_write_read_roundtrip(tmp_path) -> None:
    xyz = jnp.array(
        [[0.1, -0.2, 0.3], [1.0, 2.0, 3.0], [-4.5, 0.0, 9.25]], dtype=jnp.float32
    )
    rgb = jnp.array([[0, 128, 255], [10, 20, 30], [200, 100, 50]], dtype=jnp.uint8)

    path = save_ply_xyzrgb(tmp_path / "cloud.ply", xyz, rgb)
    xyz_read, rgb_read = load_ply_xyzrgb(path)

    np.testing.assert_allclose(np.asarray(xyz_read), np.asarray(xyz), atol=1e-5)
    np.testing.assert_array_equal(np.asarray(rgb_read), np.asarray(rgb))
    assert xyz_read.dtype == jnp.float32
    assert rgb_read.dtype == jnp.uint8


def test_load_tolerates_extra_int_properties(tmp_path) -> None:
    ply_path = tmp_path / "bench_style.ply"
    ply_path.write_text(_BENCH_STYLE_PLY, encoding="utf-8")

    xyz, rgb = load_ply_xyzrgb(ply_path)

    assert xyz.shape == (3, 3)
    np.testing.assert_allclose(
        np.asarray(xyz[0]), np.array([0.5, -1.25, 2.0]), atol=1e-6
    )
    np.testing.assert_array_equal(np.asarray(rgb[2]), np.array([70, 80, 90]))


def test_load_rejects_binary_format(tmp_path) -> None:
    ply_path = tmp_path / "binary.ply"
    ply_path.write_text(
        "ply\nformat binary_little_endian 1.0\nelement vertex 0\nend_header\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only ASCII"):
        load_ply_xyzrgb(ply_path)


def test_load_rejects_missing_color_properties(tmp_path) -> None:
    ply_path = tmp_path / "no_color.ply"
    ply_path.write_text(
        "ply\nformat ascii 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n0 0 0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="missing vertex properties"):
        load_ply_xyzrgb(ply_path)
