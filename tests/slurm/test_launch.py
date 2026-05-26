from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "scripts/slurm/configs"
# Legacy sbatch scripts were archived in s5 (3D-34); they remain the frozen
# golden references for the render-equivalence tests below.
LEGACY_SBATCH = "archiv/slurm-legacy-sbatch"


def _load_launch_module():
    spec = importlib.util.spec_from_file_location("slurm_launch", ROOT / "scripts/slurm/launch.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register before exec so pydantic can resolve the deferred (PEP 563)
    # forward refs (SbatchConfig/SmokeConfig) via sys.modules at validate time.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


launch = _load_launch_module()


def run_launch(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "scripts/slurm/launch.sh", *args],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
        env={**os.environ, **(env or {})},
    )


def training_command(text: str) -> tuple[str, str, dict[str, str]]:
    """Extract (python_cmd, script, {flag: value}) from a rendered sbatch's
    training invocation (the python line followed by `--flag value` continuations)."""

    lines = text.splitlines()
    idx = next(i for i, line in enumerate(lines) if line.rstrip().endswith(".py \\"))
    head = lines[idx].strip().rstrip(" \\")
    script = head.split()[-1]
    python_cmd = " ".join(head.split()[:-1])
    flags: dict[str, str] = {}
    for line in lines[idx + 1:]:
        stripped = line.strip().rstrip(" \\")
        if not stripped.startswith("--"):
            break
        flag, _, value = stripped.partition(" ")
        flags[flag] = value.strip().strip('"')
    return python_cmd, script, flags


# --------------------------------------------------------------------------- #
# s1 — preserved contract                                                      #
# --------------------------------------------------------------------------- #


def test_l1_vggt_dry_run_matches_legacy_sbatch() -> None:
    result = run_launch("l1_vggt", "--dry-run")

    assert result.returncode == 0, result.stderr
    expected = (ROOT / LEGACY_SBATCH / "train_curriculum_l1_vggt.sbatch").read_text()
    assert result.stdout == expected


def test_smoke_then_prod_uses_afterok_dependency_before_prod_submit(tmp_path: Path) -> None:
    calls_file = tmp_path / "sbatch-calls.txt"
    fake_sbatch = tmp_path / "sbatch"
    fake_sbatch.write_text(
        "#!/usr/bin/env bash\n"
        "cat >/dev/null\n"
        "printf '%s\\n' \"$*\" >> \"$SBATCH_CALLS\"\n"
        "wc -l < \"$SBATCH_CALLS\"\n"
    )
    fake_sbatch.chmod(0o755)

    result = run_launch(
        "l1_vggt",
        "--smoke-then-prod",
        env={"PATH": f"{tmp_path}:{os.environ['PATH']}", "SBATCH_CALLS": str(calls_file)},
    )

    assert result.returncode == 0, result.stderr
    calls = calls_file.read_text().splitlines()
    assert calls[0] == "--parsable"
    assert calls[1] == "--parsable --dependency=afterok:1 --kill-on-invalid-dep=yes"


def test_missing_required_field_fails_validation_before_sbatch(tmp_path: Path) -> None:
    """A config missing a required structural field (script) must fail fast,
    before any sbatch process is started."""

    broken = CONFIG_DIR / "broken_missing_script.yaml"
    broken.write_text(
        "extends: _base\n"
        "job_name: broken\n"
        "output_dir: output/broken\n"
        "args:\n"
        "  steps: 100\n"
    )

    fake_sbatch = tmp_path / "sbatch"
    fake_sbatch.write_text("#!/usr/bin/env bash\nexit 88\n")
    fake_sbatch.chmod(0o755)

    try:
        result = run_launch(
            "broken_missing_script",
            env={"PATH": f"{tmp_path}:{os.environ['PATH']}"},
        )
    finally:
        broken.unlink(missing_ok=True)

    assert result.returncode == 2  # validation error, not the fake sbatch's 88
    assert "script" in result.stderr
    assert "Field required" in result.stderr
    assert shutil.which("sbatch", path=str(tmp_path)) is not None


# --------------------------------------------------------------------------- #
# s2 — curriculum L2/L3/L4 via recursive `extends`                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "variant,legacy",
    [
        ("l2_vggt", f"{LEGACY_SBATCH}/train_curriculum_l2_vggt.sbatch"),
        ("l3_vggt", f"{LEGACY_SBATCH}/train_curriculum_l3_vggt.sbatch"),
        ("l4_vggt", f"{LEGACY_SBATCH}/train_curriculum_l4_vggt.sbatch"),
    ],
)
def test_curriculum_variants_match_legacy_args(variant: str, legacy: str) -> None:
    """Each L2/L3/L4 config extends l1_vggt and renders the same training
    command + identical flags as its hand-written sbatch (order-insensitive)."""

    rendered = launch.render_sbatch(launch.load_config(variant), mode="prod")
    r_python, r_script, r_flags = training_command(rendered)
    l_python, l_script, l_flags = training_command((ROOT / legacy).read_text())

    assert r_python == l_python == "uv run python"
    assert r_script == l_script
    assert r_flags == l_flags


def test_extends_chain_is_recursive() -> None:
    """l2_vggt -> l1_vggt -> _base must fully resolve (no leftover `extends`)."""

    config = launch.load_config("l2_vggt")
    # Inherited from _base via l1_vggt:
    assert config.sbatch.gres == "gpu:1"
    assert config.args["render_resolution"] == 518
    # Own override:
    assert config.script.endswith("run_jax_habitat_l2_vggt.py")
    assert config.args["val_data"] == "data/val_replay/val_200ep.npz"


def test_circular_extends_is_rejected(tmp_path: Path) -> None:
    a = CONFIG_DIR / "cycle_a.yaml"
    b = CONFIG_DIR / "cycle_b.yaml"
    a.write_text("extends: cycle_b\njob_name: a\noutput_dir: o\nscript: s.py\n")
    b.write_text("extends: cycle_a\njob_name: b\noutput_dir: o\nscript: s.py\n")
    try:
        with pytest.raises(ValueError, match="Circular extends"):
            launch.load_config("cycle_a")
    finally:
        a.unlink(missing_ok=True)
        b.unlink(missing_ok=True)


def test_sweep_submits_every_variant(tmp_path: Path) -> None:
    calls_file = tmp_path / "calls.txt"
    fake_sbatch = tmp_path / "sbatch"
    fake_sbatch.write_text(
        "#!/usr/bin/env bash\ncat >/dev/null\nprintf 'x\\n' >> \"$SBATCH_CALLS\"\nwc -l < \"$SBATCH_CALLS\"\n"
    )
    fake_sbatch.chmod(0o755)

    result = run_launch(
        "l1_vggt", "l2_vggt", "l3_vggt", "l4_vggt", "--smoke",
        env={"PATH": f"{tmp_path}:{os.environ['PATH']}", "SBATCH_CALLS": str(calls_file)},
    )

    assert result.returncode == 0, result.stderr
    assert len(calls_file.read_text().splitlines()) == 4


# --------------------------------------------------------------------------- #
# s3 — aggregator-MLP family (variable args, timestamp naming)                 #
# --------------------------------------------------------------------------- #


def test_aggregator_prod_args() -> None:
    rendered = launch.render_sbatch(launch.load_config("aggregator_mlp_v1"), mode="prod")
    python_cmd, script, flags = training_command(rendered)

    assert python_cmd == "uv run python"
    assert script.endswith("run_jax_habitat_vggt_aggregator_mlp.py")
    assert flags["--steps"] == "2000000"
    assert flags["--prefill"] == "5000"
    assert flags["--checkpoint_every"] == "50000"
    assert flags["--wandb_tags"] == "agg-mlp-prod-v1,pool-on-device,skip-heads,prod-42h"
    # No wandb_project / render_resolution for this script family.
    assert "--wandb_project" not in flags
    assert "--render_resolution" not in flags
    # Timestamp-based run id requires a TIMESTAMP definition in the script body.
    assert "TIMESTAMP=$(date +%Y%m%d-%H%M%S)" in rendered
    assert flags["--output_dir"] == "output/prod/agg-mlp-prod-v1-${TIMESTAMP}"


def test_aggregator_smoke_overrides() -> None:
    rendered = launch.render_sbatch(launch.load_config("aggregator_mlp_v1"), mode="smoke")
    _, _, flags = training_command(rendered)

    assert flags["--steps"] == "800"
    assert flags["--prefill"] == "200"
    assert "#SBATCH --time=00:20:00" in rendered
    assert "agg-mlp-fast-path-smoke" in flags["--wandb_tags"]


def test_aggregator_prod_is_strict_bash() -> None:
    rendered = launch.render_sbatch(launch.load_config("aggregator_mlp_v1"), mode="prod")
    assert "set -euo pipefail" in rendered


# --------------------------------------------------------------------------- #
# s4 — offline-buffer collector (hyphen flags, env, setup hooks)               #
# --------------------------------------------------------------------------- #


def test_offline_buffer_uses_hyphen_flags_and_venv_python() -> None:
    rendered = launch.render_sbatch(launch.load_config("offline_buffer_3d25"), mode="prod")
    python_cmd, script, flags = training_command(rendered)

    assert python_cmd == ".venv/bin/python"
    assert script.endswith("collect_offline_buffer.py")
    assert flags["--n-steps"] == "400000"
    assert flags["--collect-seed"] == "42"
    assert flags["--out-dir"] == "data/offline_buffer"
    assert flags["--skeleton-flush-every"] == "10000"
    assert flags["--checkpoint"] == "${CNN_CHECKPOINT}"


def test_offline_buffer_setup_hooks_present() -> None:
    rendered = launch.render_sbatch(launch.load_config("offline_buffer_3d25"), mode="prod")
    assert "./scripts/setup_worktree.sh" in rendered
    assert "bash scripts/slurm/hooks/link_external.sh" in rendered


def test_offline_buffer_env_override_wins_over_yaml_default() -> None:
    custom = "output/custom/run-9/checkpoints/step_x.pkl"
    config = launch.load_config("offline_buffer_3d25", env_overrides={"CNN_CHECKPOINT": custom})
    rendered = launch.render_sbatch(config, mode="smoke")
    assert f'export CNN_CHECKPOINT="{custom}"' in rendered
    # Smoke also reduces the step budget.
    _, _, flags = training_command(rendered)
    assert flags["--n-steps"] == "5000"
