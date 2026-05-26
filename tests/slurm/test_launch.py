from __future__ import annotations

import shutil
import subprocess
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run_launch(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "scripts/slurm/launch.sh", *args],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )


def test_l1_vggt_dry_run_matches_legacy_sbatch() -> None:
    result = run_launch("l1_vggt", "--dry-run")

    assert result.returncode == 0, result.stderr
    expected = (ROOT / "scripts/r2dreamer/slurm/train_curriculum_l1_vggt.sbatch").read_text()
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

    env = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "SBATCH_CALLS": str(calls_file),
    }
    result = subprocess.run(
        ["bash", "scripts/slurm/launch.sh", "l1_vggt", "--smoke-then-prod"],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    calls = calls_file.read_text().splitlines()
    assert calls[0] == "--parsable"
    assert calls[1] == "--parsable --dependency=afterok:1 --kill-on-invalid-dep=yes"


def test_external_offline_config_renders_external_venv_and_kebab_args() -> None:
    prod = run_launch("external_offline_wp_cp_seed0", "--dry-run")
    smoke = run_launch("external_offline_wp_cp_seed0", "--smoke", "--dry-run")

    assert prod.returncode == 0, prod.stderr
    assert smoke.returncode == 0, smoke.stderr

    assert "#SBATCH --partition=gpu_h100_il" in prod.stdout
    assert "#SBATCH --mem=160G" in prod.stdout
    assert "external/r2dreamer/.venv/bin/python scripts/r2dreamer/train_external_offline.py" in prod.stdout
    assert "--buffer-dir \"data/offline_buffer\"" in prod.stdout
    assert "--output-dir \"output/3d45-external-offline/wp_cp-seed0/run-${SLURM_JOB_ID}\"" in prod.stdout
    assert "--wandb \\" in prod.stdout
    assert "--wandb-project \"3d-vla-objectnav-offline-ablation\"" in prod.stdout

    assert "#SBATCH --partition=gpu_h100_short" in smoke.stdout
    assert "--buffer-dir \"data/offline_buffer_smoke\"" in smoke.stdout
    assert "--batch-size 4" in smoke.stdout
    assert "    --wandb \\" not in smoke.stdout
    assert "latest.pt missing in smoke output" in smoke.stdout
    assert "=== Smoke PASS ===" in smoke.stdout


def test_missing_steps_fails_validation_before_sbatch(tmp_path: Path) -> None:
    config_dir = ROOT / "scripts/slurm/configs"
    broken = config_dir / "missing_steps.yaml"
    broken.write_text(
        "extends: _base\n"
        "job_name: broken\n"
        "output_dir: output/broken\n"
        "script: scripts/r2dreamer/run_jax_habitat_vggt.py\n"
        "args:\n"
        "  prefill: 5000\n"
        "  checkpoint_every: 100000\n"
        "  output_dir: output/broken/run-${SLURM_JOB_ID}\n"
        "  seed: ${SLURM_JOB_ID}\n"
        "  log_every: 250\n"
        "  wandb_project: 3d-vla-objectnav\n"
        "  wandb_name: broken-${SLURM_JOB_ID}\n"
        "  wandb_tags: broken\n"
        "  render_resolution: 518\n"
    )

    fake_sbatch = tmp_path / "sbatch"
    fake_sbatch.write_text("#!/usr/bin/env bash\nexit 88\n")
    fake_sbatch.chmod(0o755)

    try:
        result = subprocess.run(
            ["bash", "scripts/slurm/launch.sh", "missing_steps"],
            cwd=ROOT,
            check=False,
            text=True,
            capture_output=True,
            env={
                **os.environ,
                "PATH": f"{tmp_path}:{os.environ['PATH']}",
            },
        )
    finally:
        broken.unlink(missing_ok=True)

    assert result.returncode == 2
    assert "args.steps" in result.stderr
    assert "Field required" in result.stderr
    assert shutil.which("sbatch", path=str(tmp_path)) is not None
