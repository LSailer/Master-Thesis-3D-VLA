"""Pytest environment guards for optional accelerator/integration tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _jax_gpu_available() -> bool:
    """Return True when JAX can see a GPU backend in this process."""
    try:
        import jax

        return any(device.platform == "gpu" for device in jax.devices())
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """Skip optional hardware/checkpoint-heavy tests on ordinary unit runs."""
    have_jax_gpu = _jax_gpu_available()
    skip_gpu = pytest.mark.skip(reason="requires a JAX GPU backend")
    skip_vggt_parity = pytest.mark.skip(
        reason="requires RUN_VGGT_PARITY=1; loads/runs the full VGGT checkpoint"
    )

    run_vggt_parity = os.environ.get("RUN_VGGT_PARITY") == "1"

    for item in items:
        if "gpu" in item.keywords and not have_jax_gpu:
            item.add_marker(skip_gpu)

        path = Path(str(item.fspath)).as_posix()
        if path.endswith("tests/vggt/test_jax_parity.py") and not run_vggt_parity:
            item.add_marker(skip_vggt_parity)
