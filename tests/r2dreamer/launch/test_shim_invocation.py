"""Subprocess smoke tests for launcher shims.

Runs `python <shim> --help` in a fresh subprocess and asserts exit 0. This
catches the class of bug where a shim fails to bootstrap sys.path and
`ModuleNotFoundError: src` surfaces only at sbatch invocation time —
pytest's own sys.path setup (via `[tool.pytest.ini_options]`) hides such
bugs from in-process import tests.

Cf. #91 (this test) and the Phase 4 fix in commit 43c0e6b.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

SHIMS = [
    "scripts/r2dreamer/run_jax_habitat.py",
    "scripts/r2dreamer/run_jax_habitat_l2.py",
    "scripts/r2dreamer/run_jax_habitat_l3.py",
    "scripts/r2dreamer/run_jax_habitat_l4.py",
    "scripts/r2dreamer/run_jax_habitat_vggt.py",
    "scripts/r2dreamer/run_jax_habitat_vggt_agg_raw_mlp.py",
    "scripts/r2dreamer/run_jax_habitat_hybrid_agg_pooled.py",
    "scripts/r2dreamer/run_jax_habitat_hybrid_agg_raw.py",
    "scripts/r2dreamer/run_jax_crafter.py",
    "scripts/r2dreamer/eval_habitat.py",
    "scripts/r2dreamer/run_parity_training.py",
    "scripts/r2dreamer/run_benchmark.py",
]


@pytest.mark.parametrize("shim_path", SHIMS)
def test_shim_help_exits_zero(shim_path):
    result = subprocess.run(
        [sys.executable, shim_path, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"{shim_path} --help exited {result.returncode}\n"
        f"stderr:\n{result.stderr.decode(errors='replace')}"
    )
