"""Filesystem path helpers for VGGT backends."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the repository root from inside the `src/vggt` package."""
    return Path(__file__).resolve().parents[2]


def infinite_vggt_src() -> Path:
    """Return the vendored InfiniteVGGT source directory."""
    return repo_root() / "external" / "InfiniteVGGT" / "src"
