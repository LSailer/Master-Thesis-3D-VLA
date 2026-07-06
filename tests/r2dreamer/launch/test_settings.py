"""Truth-table tests for the CLI-vs-shim precedence helpers.

Covers ``resolve_override`` (CLI value wins unless None) and
``resolve_required`` (same rule, but raises when the result would be None).
"""
# pylint: disable=missing-function-docstring

import pytest

from src.r2dreamer.launch.settings import resolve_override, resolve_required


@pytest.mark.parametrize(
    ("cli_value", "shim_default", "expected"),
    [
        ("cli", "shim", "cli"),  # CLI provided -> CLI wins
        (None, "shim", "shim"),  # CLI absent -> shim default
        ("cli", None, "cli"),  # only CLI provided
        (None, None, None),  # neither provided
        (0, "shim", 0),  # falsy-but-not-None CLI value still wins
        (False, "shim", False),  # False is a real value, not "absent"
    ],
)
def test_resolve_override_truth_table(cli_value, shim_default, expected):
    assert resolve_override(cli_value, shim_default) == expected


@pytest.mark.parametrize(
    ("cli_value", "shim_default", "expected"),
    [
        ("cli", "shim", "cli"),
        (None, "shim", "shim"),
        ("cli", None, "cli"),
        (0, None, 0),  # falsy value is valid and does not raise
    ],
)
def test_resolve_required_returns_resolved(cli_value, shim_default, expected):
    assert (
        resolve_required(cli_value, shim_default, name="output_dir", hint="--output_dir")
        == expected
    )


def test_resolve_required_raises_when_both_none():
    with pytest.raises(ValueError, match="output_dir must be set via --output_dir"):
        resolve_required(None, None, name="output_dir", hint="--output_dir")
