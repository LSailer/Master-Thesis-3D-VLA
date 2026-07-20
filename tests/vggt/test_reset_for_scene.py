"""Unit tests for scene-aware VGGT cache reset (``ResetMode.PERSIST_SCENE``).

These exercise only the cache save/restore/``reset_for_scene`` machinery and
the ``is_first`` wiring in ``_image_from_extract_input`` — no checkpoint, JIT,
or GPU is required. The extractor is built via ``object.__new__`` and only the
fields the reset path touches are populated, so the suite runs on the login
node under ``JAX_PLATFORMS=cpu``.
"""

from __future__ import annotations

import numpy as np

import jax.numpy as jnp

from src.environments.observation import ObservationFrame
from src.vggt.jax.feature_extractor import (
    JAXVGGTFeatureExtractor,
    ResetMode,
)


def _make_extractor(reset_mode: ResetMode) -> JAXVGGTFeatureExtractor:
    """Build a field-only extractor stub (no checkpoint/JIT/GPU).

    Only the attributes the reset path touches are populated, so the suite runs
    on the login node under ``JAX_PLATFORMS=cpu`` without loading weights.

    Args:
        reset_mode: The ``ResetMode`` to set on the stub.

    Returns:
        A ``JAXVGGTFeatureExtractor`` instance built via ``object.__new__`` with
        ``_reset_mode``, ``_scene_cache_store``, ``_current_scene_id`` and the
        four streaming cache fields initialized.
    """
    ext = object.__new__(JAXVGGTFeatureExtractor)
    ext._reset_mode = reset_mode
    ext._scene_cache_store = {}
    ext._current_scene_id = None
    ext._past_kvs_padded = None
    ext._last_scores = None
    ext._past_kvs_camera = None
    ext._frame_idx = 0
    return ext


def _populate(ext: JAXVGGTFeatureExtractor, tag: int) -> None:
    """Set the four streaming cache fields to a recognizable sentinel state.

    Args:
        ext: The extractor stub to populate.
        tag: Integer sentinel written into every field (array fills and
            ``_frame_idx``), so a later ``_assert_cache(ext, tag)`` can verify
            the state was restored unchanged.
    """
    ext._past_kvs_padded = [jnp.full((1, tag), tag, dtype=jnp.float32)]
    ext._last_scores = jnp.full((tag,), tag, dtype=jnp.float32)
    ext._past_kvs_camera = [jnp.full((1, tag), tag, dtype=jnp.float32)]
    ext._frame_idx = tag


def _assert_cache(ext: JAXVGGTFeatureExtractor, tag: int) -> None:
    """Assert the live cache fields carry sentinel state ``tag``.

    Args:
        ext: The extractor stub whose cache fields are checked.
        tag: The expected sentinel value (see ``_populate``).

    Raises:
        AssertionError: If ``_frame_idx`` or any cached array does not match.
    """
    assert ext._frame_idx == tag, f"frame_idx={ext._frame_idx}, want {tag}"
    np.testing.assert_allclose(
        np.asarray(ext._last_scores), np.full((tag,), tag, dtype=np.float32)
    )
    assert len(ext._past_kvs_padded) == 1
    np.testing.assert_allclose(
        np.asarray(ext._past_kvs_padded[0]), np.full((1, tag), tag, dtype=np.float32)
    )


def _is_reset(ext: JAXVGGTFeatureExtractor) -> bool:
    """Return True when all four streaming fields are at their post-``reset`` values.

    Args:
        ext: The extractor stub to inspect.

    Returns:
        True if ``_past_kvs_padded``, ``_last_scores``, ``_past_kvs_camera`` are
        all ``None`` and ``_frame_idx`` is 0.
    """
    return (
        ext._past_kvs_padded is None
        and ext._last_scores is None
        and ext._past_kvs_camera is None
        and ext._frame_idx == 0
    )


class TestResetForScenePersist:
    """``ResetMode.PERSIST_SCENE`` save/restore behaviour."""

    def test_first_scene_is_fresh_reset(self):
        ext = _make_extractor(ResetMode.PERSIST_SCENE)
        _populate(ext, 5)  # leftover state before any scene
        ext.reset_for_scene("A")
        assert ext._current_scene_id == "A"
        assert _is_reset(ext)
        assert not ext._scene_cache_store

    def test_save_restore_roundtrip_across_scenes(self):
        ext = _make_extractor(ResetMode.PERSIST_SCENE)
        # Episode 1 of scene A -> fresh reset, then stream to state 7.
        ext.reset_for_scene("A")
        _populate(ext, 7)
        # Switch to scene B: A's cache(7) is saved, B starts fresh.
        ext.reset_for_scene("B")
        assert "A" in ext._scene_cache_store
        assert ext._scene_cache_store["A"].frame_idx == 7
        assert ext._current_scene_id == "B"
        assert _is_reset(ext)
        _populate(ext, 9)  # stream B to state 9
        # Back to scene A: B's cache(9) is saved, A's cache(7) is restored.
        ext.reset_for_scene("A")
        assert ext._scene_cache_store["B"].frame_idx == 9
        assert ext._current_scene_id == "A"
        _assert_cache(ext, 7)
        # Back to B: A re-saved (still 7), B(9) restored.
        ext.reset_for_scene("B")
        assert ext._scene_cache_store["A"].frame_idx == 7
        _assert_cache(ext, 9)

    def test_full_mode_wipes_and_does_not_persist(self):
        ext = _make_extractor(ResetMode.FULL)
        _populate(ext, 4)
        ext.reset_for_scene("A")
        assert _is_reset(ext)
        assert not ext._scene_cache_store  # nothing saved under FULL
        # The FULL path does not update _current_scene_id.
        assert ext._current_scene_id is None


class TestImageFromExtractInputWiring:
    """The ``is_first`` frame must trigger a scene-aware reset, not a bare one."""

    @staticmethod
    def _frame(is_first: bool, scene_id: str) -> ObservationFrame:
        """Build a minimal ``ObservationFrame`` for the wiring tests.

        Args:
            is_first: The ``is_first`` flag to set on the frame.
            scene_id: The ``scene_id`` to set on the frame.

        Returns:
            An ``ObservationFrame`` with a zero 518x518 CHW image and the given
            ``is_first``/``scene_id``.
        """
        return ObservationFrame(
            image=np.zeros((3, 518, 518), dtype=np.uint8),
            is_first=is_first,
            scene_id=scene_id,
        )

    def test_is_first_calls_reset_for_scene_with_scene_id(self, monkeypatch):
        ext = _make_extractor(ResetMode.PERSIST_SCENE)
        calls: list[str] = []
        monkeypatch.setattr(ext, "reset_for_scene", lambda sid: calls.append(sid))
        ext._image_from_extract_input(self._frame(True, "house-42"))
        assert calls == ["house-42"]

    def test_is_first_blank_scene_id_falls_back_to_scene_key(self, monkeypatch):
        ext = _make_extractor(ResetMode.PERSIST_SCENE)
        calls: list[str] = []
        monkeypatch.setattr(ext, "reset_for_scene", lambda sid: calls.append(sid))
        # scene_id="" must fall back to "scene", matching the buffer's keying.
        ext._image_from_extract_input(self._frame(True, ""))
        assert calls == ["scene"]

    def test_non_first_frame_does_not_reset(self, monkeypatch):
        ext = _make_extractor(ResetMode.PERSIST_SCENE)
        calls: list[str] = []
        monkeypatch.setattr(ext, "reset_for_scene", lambda sid: calls.append(sid))
        ext._image_from_extract_input(self._frame(False, "house-42"))
        assert not calls
