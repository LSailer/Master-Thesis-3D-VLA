"""Unit tests for launch/train._effective_curriculum_inputs."""

from types import SimpleNamespace

import pytest

from src.r2dreamer.launch.train import _effective_curriculum_inputs

REG = {"habitat": object(), "crafter": object()}


def _call(env, *, curriculum=None, args_curriculum=None, curriculum_path=None):
    args = SimpleNamespace(curriculum=args_curriculum, curriculum_path=curriculum_path)
    return _effective_curriculum_inputs(
        env=env, args=args, curriculum=curriculum, env_registry=REG
    )


def test_habitat_kwarg_curriculum():
    assert _call("habitat", curriculum="L1") == ("L1", None)


def test_cli_curriculum_wins_over_kwarg():
    assert _call("habitat", curriculum="L1", args_curriculum="L2") == ("L2", None)


def test_habitat_path_only():
    assert _call("habitat", curriculum_path="/tmp/c.json") == (None, "/tmp/c.json")


def test_crafter_ok_without_curriculum():
    assert _call("crafter") == (None, None)


def test_habitat_requires_curriculum():
    with pytest.raises(ValueError, match="Habitat env requires"):
        _call("habitat")


def test_crafter_rejects_curriculum():
    with pytest.raises(ValueError, match="Crafter env does not"):
        _call("crafter", curriculum="L1")


def test_unknown_env():
    with pytest.raises(KeyError, match="Unknown env"):
        _call("unknown")
