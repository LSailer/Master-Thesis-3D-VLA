"""Shared resolution helpers for the launcher entry points.

DRY: `train()` and `evaluate()` both turn a ``--curriculum_path`` CLI escape
hatch plus an optional named curriculum into a concrete JSON path. Keeping that
logic here means a change to the curriculum-resolution rule is made once, not
synced across two entry points. Per-env validation (habitat requires a
curriculum; crafter forbids one) intentionally stays in the callers.
"""

from __future__ import annotations


def resolve_curriculum_path(
    curriculum_path_arg: str | None, curriculum: str | None
) -> str | None:
    """Resolve a curriculum JSON path from the CLI arg or a registry name.

    ``--curriculum_path`` (``curriculum_path_arg``) is the explicit escape hatch and
    wins when set. Otherwise a named ``curriculum`` (e.g. ``"L1"``) is looked up in
    the ``CURRICULA`` registry. Returns ``None`` when neither is supplied; the caller
    applies the env-specific requirement.
    """
    from src.r2dreamer.launch.curricula import CURRICULA

    if curriculum_path_arg is not None:
        return curriculum_path_arg
    if curriculum is not None:
        if curriculum not in CURRICULA:
            raise KeyError(
                f"Unknown curriculum {curriculum!r}. Available: {list(CURRICULA)}"
            )
        return str(CURRICULA[curriculum])
    return None
