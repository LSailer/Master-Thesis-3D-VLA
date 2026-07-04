"""CPU tests for the VGGT-side PLY point-cloud snapshot writer.

Covers ``JAXVGGTFeatureExtractor._write_ascii_ply_xyzrgb`` (the file-boundary
helper behind ``write_point_cloud_ply``) — round-trips with the existing
``load_ascii_ply_xyzrgb`` reader. The full ``write_point_cloud_ply`` path runs
the real point head and is exercised by the SLURM smoke (GPU + weights).
"""

import numpy as np
import pytest

from src.r2dreamer.observation_preparation.static_house_context import (
    load_ascii_ply_xyzrgb,
)
from src.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor


def _writer():
    # Static method — no extractor construction (avoids GPU/weights).
    return JAXVGGTFeatureExtractor._write_ascii_ply_xyzrgb


def test_ply_writer_round_trips_with_reader(tmp_path):
    xyz = np.array(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.5, 0.25, 4.75]], dtype=np.float32
    )
    rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    path = str(tmp_path / "house.ply")

    _writer()(path, xyz, rgb)

    loaded = load_ascii_ply_xyzrgb(path)
    assert loaded.shape == (3, 6)
    np.testing.assert_allclose(loaded[:, :3], xyz, atol=1e-5)
    # RGB is stored as uchar and read back as float32.
    np.testing.assert_allclose(loaded[:, 3:], rgb.astype(np.float32))


def test_ply_writer_accepts_float_rgb(tmp_path):
    xyz = np.zeros((2, 3), dtype=np.float32)
    rgb = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=np.float32)
    path = str(tmp_path / "house_float.ply")

    _writer()(path, xyz, rgb)

    loaded = load_ascii_ply_xyzrgb(path)
    np.testing.assert_allclose(loaded[:, 3:], (rgb * 255).astype(np.uint8).astype(np.float32))


def test_ply_writer_creates_parent_dirs(tmp_path):
    path = str(tmp_path / "nested" / "dir" / "house.ply")
    _writer()(path, np.zeros((1, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.uint8))
    import os

    assert os.path.exists(path)


def test_ply_writer_header_has_required_fields(tmp_path):
    path = str(tmp_path / "house.ply")
    _writer()(path, np.zeros((1, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.uint8))
    with open(path, encoding="ascii") as handle:
        header = "".join(handle.readline() for _ in range(10))
    for field in ("ply", "format ascii 1.0", "element vertex 1", "end_header"):
        assert field in header
    for prop in ("x", "y", "z", "red", "green", "blue"):
        assert f"property float {prop}" in header or f"property uchar {prop}" in header


def test_write_point_cloud_ply_raises_before_any_extract(monkeypatch):
    """Guard the contract: calling write_point_cloud_ply before extract() raises."""
    # Build a shell instance without running __init__ (avoids GPU/weights).
    extractor = JAXVGGTFeatureExtractor.__new__(JAXVGGTFeatureExtractor)
    extractor._last_out_list = None
    extractor._last_images = None
    with pytest.raises(RuntimeError, match="before any extract"):
        extractor.write_point_cloud_ply("/tmp/should_not_exist.ply")