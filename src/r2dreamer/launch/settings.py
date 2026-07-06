"""Shared CLI-vs-shim precedence helpers for the launch entry points.

Both ``train()`` and ``evaluate()`` accept shim-supplied keyword defaults
(``output_dir``, ``checkpoint``, ...) that a CLI flag overrides when the user
passes it explicitly. argparse encodes "not provided" as ``None`` for these
flags, so the precedence rule is uniformly "CLI value wins unless it is None".
This module is the single home for that rule.
"""

from __future__ import annotations

from typing import Any, TypeVar

_T = TypeVar("_T")


def resolve_override(cli_value: _T | None, shim_default: _T | None) -> _T | None:
    """Resolve one CLI-flag / shim-kwarg pair by the launch precedence rule.

    argparse stores "flag not provided" as ``None`` for the overridable flags,
    so a non-None CLI value always wins over the shim-supplied default.

    Args:
      cli_value: The value parsed from argparse (``None`` when not provided).
      shim_default: The default supplied by the calling shim.

    Returns:
      ``cli_value`` when it is not ``None``, otherwise ``shim_default``.
    """
    return cli_value if cli_value is not None else shim_default


def resolve_required(
    cli_value: _T | None, shim_default: _T | None, *, name: str, hint: str
) -> _T:
    """Resolve an override that must end up non-None, raising otherwise.

    Args:
      cli_value: The value parsed from argparse (``None`` when not provided).
      shim_default: The default supplied by the calling shim.
      name: The setting name, used in the error message.
      hint: How the caller can supply the value, used in the error message.

    Returns:
      The resolved value (guaranteed non-None).

    Raises:
      ValueError: If both ``cli_value`` and ``shim_default`` are ``None``.
    """
    resolved = resolve_override(cli_value, shim_default)
    if resolved is None:
        raise ValueError(f"{name} must be set via {hint}")
    return resolved


def resolve_eval_settings(
    args: Any, *, encoder: str, checkpoint: str | None, output_dir: str | None
) -> tuple[str, str | None, str]:
    """Resolve (encoder, checkpoint, output_dir) for evaluate() with precedence.

    CLI flags override the shim-supplied kwargs. ``checkpoint`` is required
    unless ``--random`` is set; ``output_dir`` is always required.

    Args:
      args: Parsed argparse namespace (needs ``encoder``, ``checkpoint``,
        ``output_dir``, ``random`` attributes).
      encoder: Shim-supplied encoder default.
      checkpoint: Shim-supplied checkpoint default.
      output_dir: Shim-supplied output-dir default.

    Returns:
      A ``(eff_encoder, eff_checkpoint, eff_output_dir)`` tuple.

    Raises:
      ValueError: If checkpoint is unset while not ``--random``, or if
        output_dir is unset.
    """
    eff_encoder = resolve_override(args.encoder, encoder)

    eff_checkpoint = resolve_override(args.checkpoint, checkpoint)
    if not args.random and eff_checkpoint is None:
        raise ValueError(
            "checkpoint must be set via evaluate(..., checkpoint=...) or --checkpoint"
        )

    eff_output_dir = resolve_required(
        args.output_dir,
        output_dir,
        name="output_dir",
        hint="evaluate(..., output_dir=...) or --output_dir",
    )
    return eff_encoder, eff_checkpoint, eff_output_dir
