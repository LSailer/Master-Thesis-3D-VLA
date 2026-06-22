"""Subprocess smoke tests for launcher entrypoints.

Runs `python <entrypoint> ... --help` in a fresh subprocess and asserts exit 0.
This catches the class of bug where an entrypoint fails to bootstrap sys.path
and `ModuleNotFoundError: src` surfaces only at sbatch invocation time —
pytest's own sys.path setup (via `[tool.pytest.ini_options]`) hides such bugs
from in-process import tests.

The per-run `run_jax_*.py` shims were replaced by the single `run.py` dispatcher
(run id positional). Every RUN_CONFIGS run is smoked through it here, which also
exercises `launch_run`'s per-run encoder validation. Cf. #91 (this test) and the
Phase 4 fix in commit 43c0e6b.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_run_configs() -> ModuleType:
    """Load scripts/r2dreamer/_run_configs.py without mutating sys.path."""
    spec = importlib.util.spec_from_file_location(
        "_run_configs", REPO_ROOT / "scripts" / "r2dreamer" / "_run_configs.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("Could not load scripts/r2dreamer/_run_configs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUN_IDS = sorted(cast(dict[str, object], getattr(_load_run_configs(), "RUN_CONFIGS")))

# Entrypoints that remain standalone, i.e. not folded into the run.py dispatcher.
STANDALONE_SHIMS = ["scripts/r2dreamer/eval_habitat.py"]


def _assert_help_exits_zero(argv: list[str]) -> None:
    result = subprocess.run(
        [sys.executable, *argv, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        f"{' '.join(argv)} --help exited {result.returncode}\n"
        f"stderr:\n{result.stderr.decode(errors='replace')}"
    )


@pytest.mark.parametrize("run_id", RUN_IDS)
def test_dispatcher_run_help_exits_zero(run_id: str) -> None:
    """Every configured run supports --help through the dispatcher."""
    _assert_help_exits_zero(["scripts/r2dreamer/run.py", run_id])


@pytest.mark.parametrize("shim_path", STANDALONE_SHIMS)
def test_standalone_shim_help_exits_zero(shim_path: str) -> None:
    """Standalone shims support --help from a subprocess."""
    _assert_help_exits_zero([shim_path])
