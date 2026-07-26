"""Unit tests for src.main._effective_curriculum.

The env registry is gone: ``src.main`` validates against its own ``ENVS`` tuple,
and the function returns just the effective curriculum level. A
``--curriculum_path`` only satisfies the "habitat needs a curriculum" rule; it is
not echoed back.
"""

from types import SimpleNamespace

import pytest

from src.main import _build_parser, _effective_curriculum
from src.r2dreamer.launch.parser import _build_parser_train


def _call(env, *, curriculum=None, args_curriculum=None, curriculum_path=None):
    args = SimpleNamespace(curriculum=args_curriculum, curriculum_path=curriculum_path)
    return _effective_curriculum(env=env, args=args, curriculum=curriculum)


def _from_cli(*argv):
    """Resolve the curriculum the way ``main()`` -> ``train()`` does, no env."""
    top, rest = _build_parser().parse_known_args(list(argv))
    return _effective_curriculum(
        env=top.env,
        args=_build_parser_train().parse_args(rest),
        curriculum=top.curriculum,
    )


def test_habitat_kwarg_curriculum():
    assert _call("habitat", curriculum="L1") == "L1"


def test_cli_curriculum_wins_over_kwarg():
    assert _call("habitat", curriculum="L1", args_curriculum="L2") == "L2"


def test_habitat_path_only():
    # A curriculum file satisfies the requirement without naming a level.
    assert _call("habitat", curriculum_path="/tmp/c.json") is None


def test_crafter_ok_without_curriculum():
    assert _call("crafter") is None


def test_habitat_requires_curriculum():
    with pytest.raises(ValueError, match="Habitat env requires"):
        _call("habitat")


def test_crafter_rejects_curriculum():
    with pytest.raises(ValueError, match="Crafter env does not"):
        _call("crafter", curriculum="L1")


def test_crafter_rejects_a_curriculum_path_too():
    with pytest.raises(ValueError, match="Crafter env does not"):
        _call("crafter", curriculum_path="/tmp/c.json")


def test_crafter_is_reachable_from_the_bare_cli():
    """``train --env crafter`` must parse to no curriculum at all.

    A non-None parser default for ``--curriculum`` made this env unreachable:
    every crafter invocation raised, and no CLI value could clear the default.
    """
    assert _from_cli("train", "--env", "crafter") is None


def test_habitat_from_the_bare_cli_still_requires_a_curriculum():
    """Guards the test above: dropping the default must not un-guard habitat."""
    with pytest.raises(ValueError, match="Habitat env requires"):
        _from_cli("train", "--env", "habitat")


def test_habitat_curriculum_from_the_cli_reaches_the_run():
    assert _from_cli("train", "--env", "habitat", "--curriculum", "L3") == "L3"


def test_unknown_env():
    with pytest.raises(KeyError, match="Unknown env"):
        _call("unknown")
