"""Pytest environment guards for optional accelerator/integration tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


def _jax_gpu_available() -> bool:
    """Return True when JAX can see a GPU backend in this process."""
    try:
        import jax

        return any(device.platform == "gpu" for device in jax.devices())
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _release_agent_jit_caches():
    """Drop JAX's jit caches after any test that built an R2DreamerAgent.

    ``R2DreamerAgent.act`` and ``.train_step`` are jitted with
    ``static_argnums=(0,)``, so every agent instance is part of a cache key and
    JAX holds it - plus its params and its compiled executable - alive for the
    rest of the process. Production builds one agent per run and does not care;
    a test session builds dozens, and without this the CPU backend aborts
    mid-compile once the retained executables pile up (~7 GB RSS).

    ``jax.clear_caches`` is the only thing that releases them: the jitted
    function's own ``clear_cache`` leaves the instances referenced. It is
    global, so it only fires for tests that actually put an agent in the cache
    - those recompile per instance anyway, so nothing reusable is thrown away.
    """
    yield
    agent_module = sys.modules.get("src.r2dreamer.agent")
    if agent_module is None:
        return
    agent_class = agent_module.R2DreamerAgent
    if any(
        method._cache_size() for method in (agent_class.act, agent_class.train_step)
    ):
        sys.modules["jax"].clear_caches()


def pytest_collection_modifyitems(config, items):
    """Skip optional hardware/checkpoint-heavy tests on ordinary unit runs."""
    have_jax_gpu = _jax_gpu_available()
    skip_gpu = pytest.mark.skip(reason="requires a JAX GPU backend")
    skip_vggt_parity = pytest.mark.skip(
        reason="requires RUN_VGGT_PARITY=1; loads/runs the full VGGT checkpoint"
    )

    run_vggt_parity = os.environ.get("RUN_VGGT_PARITY") == "1"
    run_habitat_e2e = os.environ.get("RUN_HABITAT_E2E") == "1"
    skip_habitat_e2e = pytest.mark.skip(
        reason="requires RUN_HABITAT_E2E=1; loads Habitat-Sim + HM3D scenes"
    )

    for item in items:
        if "gpu" in item.keywords and not have_jax_gpu:
            item.add_marker(skip_gpu)

        if "habitat_sim" in item.keywords and not run_habitat_e2e:
            item.add_marker(skip_habitat_e2e)

        path = Path(str(item.fspath)).as_posix()
        if path.endswith("tests/vggt/test_jax_parity.py") and not run_vggt_parity:
            item.add_marker(skip_vggt_parity)
