"""Tests for shared video helpers."""
import numpy as np
import pytest

from src.shared.video_utils import write_frames_mp4


def test_empty_frames_returns_none(tmp_path):
    assert write_frames_mp4([], tmp_path / "out.mp4") is None


def test_writes_mp4_from_hwc_uint8(tmp_path):
    pytest.importorskip("moviepy")
    rng = np.random.default_rng(0)
    frames = [
        rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8) for _ in range(8)
    ]
    out = write_frames_mp4(frames, tmp_path / "sub" / "out.mp4", fps=4)
    assert out is not None and out.exists() and out.stat().st_size > 0


def test_accepts_chw_and_jax_arrays(tmp_path):
    pytest.importorskip("moviepy")
    jnp = pytest.importorskip("jax.numpy")
    frames = [
        np.zeros((3, 32, 32), dtype=np.uint8),  # CHW host
        jnp.full((32, 32, 3), 128, dtype=jnp.uint8),  # HWC device
    ]
    out = write_frames_mp4(frames, tmp_path / "mixed.mp4", fps=2)
    assert out is not None and out.exists()
