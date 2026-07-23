"""CPU tests for the shared PLY writer and the extractor dump entry point.

The writer (``shared.ply_io.write_world_points_ply``) packs binary PLY
records on device in JAX; round-trips use ``open3d.io.read_point_cloud``
(Open3D remains the reader). The full
``write_point_cloud_ply`` path runs the real point head and is exercised by
the SLURM smoke (GPU + weights).
"""

import numpy as np
import open3d as o3d
import pytest

from src.shared.ply_io import write_world_points_ply
from src.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor


def _read_xyz_rgb(path):
    pcd = o3d.io.read_point_cloud(str(path))
    return np.asarray(pcd.points), np.rint(np.asarray(pcd.colors) * 255.0)


def test_ply_writer_round_trips_via_o3d(tmp_path):
    xyz = np.array(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-1.5, 0.25, 4.75]], dtype=np.float32
    )
    rgb = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    path = tmp_path / "house.ply"

    write_world_points_ply(path, xyz, rgb)

    xyz_back, rgb_back = _read_xyz_rgb(path)
    np.testing.assert_allclose(xyz_back, xyz, atol=1e-5)
    np.testing.assert_allclose(rgb_back, rgb.astype(np.float64), atol=1.0)


def test_ply_writer_flattens_point_map_and_batch_dims(tmp_path):
    xyz = np.zeros((1, 4, 4, 3), dtype=np.float32)  # batched point map
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)  # HWC image
    path = tmp_path / "map.ply"

    write_world_points_ply(path, xyz, rgb)

    xyz_back, _ = _read_xyz_rgb(path)
    assert xyz_back.shape == (16, 3)


def test_ply_writer_rejects_chw_image(tmp_path):
    xyz = np.zeros((4, 4, 3), dtype=np.float32)
    chw = np.zeros((3, 4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="transpose to HWC"):
        write_world_points_ply(tmp_path / "bad.ply", xyz, chw)


def test_ply_writer_rejects_mismatched_vertex_counts(tmp_path):
    xyz = np.zeros((5, 3), dtype=np.float32)
    rgb = np.zeros((4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="vertex count"):
        write_world_points_ply(tmp_path / "bad.ply", xyz, rgb)


def test_ply_writer_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "dir" / "house.ply"
    write_world_points_ply(
        path, np.zeros((1, 3), dtype=np.float32), np.zeros((1, 3), dtype=np.uint8)
    )
    assert path.exists()


def test_write_point_cloud_ply_raises_before_any_extract():
    """Guard the contract: calling write_point_cloud_ply before extract() raises."""
    # Build a shell instance without running __init__ (avoids GPU/weights).
    extractor = JAXVGGTFeatureExtractor.__new__(JAXVGGTFeatureExtractor)
    extractor._last_out_list = None
    extractor._last_images = None
    with pytest.raises(RuntimeError, match="before any extract"):
        extractor.write_point_cloud_ply("/tmp/should_not_exist.ply")
