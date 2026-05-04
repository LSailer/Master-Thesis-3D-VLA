"""Unit tests for build_agent_state — pure-function, no Habitat needed."""

import numpy as np
import pytest

from modules.envs.habitat_r2dreamer import _AGENT_STATE_DIM, build_agent_state


def test_shape_and_dtype():
    state = build_agent_state(
        position=np.array([0.0, 1.5, 0.0]),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        hfov_deg=90.0, height=64, width=64,
    )
    assert state.shape == (_AGENT_STATE_DIM,)
    assert state.dtype == np.float32


def test_identity_quaternion_gives_identity_rotation():
    state = build_agent_state(
        position=np.zeros(3),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        hfov_deg=90.0, height=64, width=64,
    )
    extr = state[:16].reshape(4, 4)
    np.testing.assert_allclose(extr[:3, :3], np.eye(3), atol=1e-6)


def test_translation_is_packed():
    pos = np.array([1.5, -2.0, 3.25])
    state = build_agent_state(
        position=pos, quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        hfov_deg=90.0, height=64, width=64,
    )
    extr = state[:16].reshape(4, 4)
    np.testing.assert_allclose(extr[:3, 3], pos, atol=1e-6)
    assert extr[3, 3] == 1.0


def test_rotation_block_is_orthogonal():
    state = build_agent_state(
        position=np.zeros(3),
        quat_xyzw=np.array([0.5, 0.5, 0.5, 0.5]),  # 120 deg around (1,1,1)
        hfov_deg=90.0, height=64, width=64,
    )
    R = state[:16].reshape(4, 4)[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)


def test_intrinsics_match_hfov():
    H, W, hfov = 64, 64, 90.0
    state = build_agent_state(
        position=np.zeros(3), quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        hfov_deg=hfov, height=H, width=W,
    )
    K = state[16:].reshape(3, 3)
    expected_fx = (W / 2.0) / np.tan(np.radians(hfov) / 2.0)
    assert K[0, 0] == pytest.approx(expected_fx, abs=1e-5)
    assert K[1, 1] == pytest.approx(expected_fx, abs=1e-5)  # square pixels
    assert K[0, 2] == pytest.approx(W / 2.0)
    assert K[1, 2] == pytest.approx(H / 2.0)
    assert K[2, 2] == 1.0


def test_zero_norm_quaternion_raises():
    with pytest.raises(ValueError, match="zero norm"):
        build_agent_state(
            position=np.zeros(3),
            quat_xyzw=np.zeros(4),
            hfov_deg=90.0, height=64, width=64,
        )


def test_non_unit_quaternion_normalises():
    state = build_agent_state(
        position=np.zeros(3),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 2.0]),  # 2x the identity
        hfov_deg=90.0, height=64, width=64,
    )
    R = state[:16].reshape(4, 4)[:3, :3]
    np.testing.assert_allclose(R, np.eye(3), atol=1e-6)
