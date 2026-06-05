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


def training_command(text: str) -> tuple[str, str, str | None, dict[str, str]]:
    """Extract (python_cmd, script, run_id, {flag: value}) from a rendered
    sbatch's training invocation (the python line followed by `--flag value`
    continuations).

    ``run_id`` is the optional leading positional emitted for dispatcher
    entrypoints (``run.py``), else ``None``. The invocation head is the first
    backslash-continued line that names a ``.py`` script (the curriculum-check
    ``generate_curriculum.py`` guard line has no trailing backslash)."""

    lines = text.splitlines()
    idx = next(
        i for i, line in enumerate(lines)
        if line.rstrip().endswith(" \\") and ".py" in line
    )
    head = lines[idx].strip().rstrip(" \\")
    tokens = head.split()
    s = next(j for j, tok in enumerate(tokens) if tok.endswith(".py"))
    python_cmd = " ".join(tokens[:s])
    script = tokens[s]
    run_id = tokens[s + 1] if len(tokens) > s + 1 else None
    flags: dict[str, str] = {}
    for line in lines[idx + 1:]:
        stripped = line.strip().rstrip(" \\")
        if not stripped.startswith("--"):
            break
        flag, _, value = stripped.partition(" ")
        flags[flag] = value.strip().strip('"')
    return python_cmd, script, run_id, flags


# --------------------------------------------------------------------------- #
# s1 — preserved contract                                                      #
# --------------------------------------------------------------------------- #


def test_l1_vggt_dry_run_matches_legacy_sbatch() -> None:
    result = run_launch("l1_vggt", "--dry-run")

    assert result.returncode == 0, result.stderr
    expected = (ROOT / LEGACY_SBATCH / "train_curriculum_l1_vggt.sbatch").read_text()
    # _base / l1_vggt have since gained three intentional changes the frozen
    # legacy script predates: (1) the GL-teardown hard-exit env var (so every
    # habitat variant exits 0 on completion), (2) multi-partition auto-select
    # (gpu_h100_il,gpu_h100), and (3) the scalars-only flags video_log_every=0 /
    # val_every=0 (videos + eval regenerated from checkpoints, not during
    # training). Normalise all three back out before the byte-equality check so
    # the rest of the contract still holds.
    rendered = result.stdout.replace('export R2DREAMER_HARD_EXIT_ON_FINISH="1"\n\n', "", 1)
    # The entrypoint migrated to the single run.py dispatcher (run id positional);
    # the frozen legacy sbatch still names the old per-run shim. Map it back.
    rendered = rendered.replace(
        "scripts/r2dreamer/run.py habitat-l1-vggt",
        "scripts/r2dreamer/run_jax_habitat_vggt.py",
        1,
    )
    rendered = rendered.replace(
        "#SBATCH --partition=gpu_h100_il,gpu_h100", "#SBATCH --partition=gpu_h100", 1
    )
    rendered = rendered.replace(
        "    --render_resolution 518 \\\n"
        "    --video_log_every 0 \\\n"
        "    --val_every 0",
        "    --render_resolution 518",
        1,
    )
    assert rendered == expected


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
    """A config missing a required structural field (job_name) must fail fast,
    before any sbatch process is started.

    (``script`` is now supplied by ``_base`` — the shared run.py dispatcher — so
    ``job_name`` is the structural field a leaf can still omit.)"""

    broken = CONFIG_DIR / "broken_missing_job_name.yaml"
    broken.write_text(
        "extends: _base\n"
        "output_dir: output/broken\n"
        "args:\n"
        "  steps: 100\n"
    )

    fake_sbatch = tmp_path / "sbatch"
    fake_sbatch.write_text("#!/usr/bin/env bash\nexit 88\n")
    fake_sbatch.chmod(0o755)

    try:
        result = run_launch(
            "broken_missing_job_name",
            env={"PATH": f"{tmp_path}:{os.environ['PATH']}"},
        )
    finally:
        broken.unlink(missing_ok=True)

    assert result.returncode == 2  # validation error, not the fake sbatch's 88
    assert "job_name" in result.stderr
    assert "Field required" in result.stderr
    assert shutil.which("sbatch", path=str(tmp_path)) is not None


# --------------------------------------------------------------------------- #
# s2 — curriculum L2/L3/L4 via recursive `extends`                             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "variant,run_id,legacy",
    [
        ("l2_vggt", "habitat-l2-vggt", f"{LEGACY_SBATCH}/train_curriculum_l2_vggt.sbatch"),
        ("l3_vggt", "habitat-l3-vggt", f"{LEGACY_SBATCH}/train_curriculum_l3_vggt.sbatch"),
        ("l4_vggt", "habitat-l4-vggt", f"{LEGACY_SBATCH}/train_curriculum_l4_vggt.sbatch"),
    ],
)
def test_curriculum_variants_match_legacy_args(variant: str, run_id: str, legacy: str) -> None:
    """Each L2/L3/L4 config extends l1_vggt and renders identical training flags
    to its hand-written sbatch (order-insensitive). The entrypoint itself
    migrated from a per-run shim to the run.py dispatcher, so the run is now
    selected by the run_id positional rather than the script path."""

    rendered = launch.render_sbatch(launch.load_config(variant), mode="prod")
    r_python, r_script, r_run_id, r_flags = training_command(rendered)
    l_python, l_script, _l_run_id, l_flags = training_command((ROOT / legacy).read_text())

    assert r_python == l_python == "uv run python"
    assert r_script.endswith("run.py")
    assert r_run_id == run_id

    # Intentional divergence from the frozen legacy sbatch: the VGGT arm is now
    # the flatten/WP-CP readout (`mlp_layers=0`) and scalars-only — video +
    # in-run eval disabled (video_log_every=0 / val_every=0, inherited from
    # l1_vggt), and the stale replay-buffer val flags (val_data /
    # val_loss_every) dropped since the parser rejects them. Videos + eval
    # metrics are regenerated from checkpoints. Normalise those out, then assert
    # the rest of the training command still matches.
    assert r_flags["--mlp_layers"] == "0"
    assert "--mlp_layers" not in l_flags
    assert r_flags["--wandb_name"] == l_flags["--wandb_name"].replace(
        "-${SLURM_JOB_ID}",
        "-flatten-${SLURM_JOB_ID}",
    )
    assert r_flags["--wandb_tags"] == f"{l_flags['--wandb_tags']},flatten,wp-cp"
    assert r_flags["--video_log_every"] == "0"
    assert r_flags["--val_every"] == "0"
    assert "--val_data" not in r_flags
    assert "--val_loss_every" not in r_flags

    intentional = {
        "--mlp_layers",
        "--wandb_name",
        "--wandb_tags",
        "--video_log_every",
        "--val_every",
        "--val_data",
        "--val_loss_every",
    }
    r_common = {k: v for k, v in r_flags.items() if k not in intentional}
    l_common = {k: v for k, v in l_flags.items() if k not in intentional}
    assert r_common == l_common


def test_extends_chain_is_recursive() -> None:
    """l2_vggt -> l1_vggt -> _base must fully resolve (no leftover `extends`)."""

    config = launch.load_config("l2_vggt")
    # Inherited from _base via l1_vggt:
    assert config.sbatch.gres == "gpu:1"
    assert config.args["render_resolution"] == 518
    # Scalars-only flags inherited from l1_vggt (videos/eval regenerated offline):
    assert config.args["video_log_every"] == 0
    assert config.args["val_every"] == 0
    # script is inherited from _base (the shared run.py dispatcher); the run is
    # selected by run_id, which l2_vggt sets as its own override:
    assert config.script.endswith("run.py")
    assert config.run_id == "habitat-l2-vggt"
    # The stale replay-buffer val flags were dropped (parser no longer accepts them):
    assert "val_data" not in config.args
    assert "val_loss_every" not in config.args


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
    python_cmd, script, run_id, flags = training_command(rendered)

    assert python_cmd == "uv run python"
    assert script.endswith("run.py")
    assert run_id == "habitat-l1-vggt-aggregator-mlp"
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
    _, _, _, flags = training_command(rendered)

    assert flags["--steps"] == "800"
    assert flags["--prefill"] == "200"
    assert "#SBATCH --time=00:20:00" in rendered
    assert "agg-mlp-fast-path-smoke" in flags["--wandb_tags"]


def test_aggregator_prod_is_strict_bash() -> None:
    rendered = launch.render_sbatch(launch.load_config("aggregator_mlp_v1"), mode="prod")
    assert "set -euo pipefail" in rendered


@pytest.mark.parametrize(
    "variant,run_id,capacity,tag",
    [
        ("l1_cnn_cap100k", "habitat-l1-cnn", "100000", "cap-100k"),
        ("l1_cnn_cap10k", "habitat-l1-cnn", "10000", "cap-10k"),
        ("l1_vggt_wpcp37_cap500k", "habitat-l1-vggt", "500000", "cap-500k"),
        ("l1_vggt_wpcp64_cap500k", "habitat-l1-vggt-wp-cp-64", "500000", "cap-500k"),
        ("l1_agg_mlp_cap500k", "habitat-l1-vggt-aggregator-mlp", "500000", "cap-500k"),
    ],
)
def test_l1_replay_capacity_ablation_configs(
    variant: str, run_id: str, capacity: str, tag: str,
) -> None:
    rendered = launch.render_sbatch(launch.load_config(variant), mode="prod")
    _, script, rendered_run_id, flags = training_command(rendered)

    assert script.endswith("run.py")
    assert rendered_run_id == run_id
    assert flags["--seed"] == "42"
    assert flags["--buffer_capacity"] == capacity
    assert tag in flags["--wandb_tags"]
    assert "3d-63" in flags["--wandb_tags"]
    assert flags["--video_log_every"] == "0"
    assert flags["--val_every"] == "0"


def test_l1_aggregator_capacity_ablation_keeps_encoder_batch_defaults() -> None:
    rendered = launch.render_sbatch(launch.load_config("l1_agg_mlp_cap500k"), mode="prod")
    _, _, _, flags = training_command(rendered)

    assert flags["--buffer_capacity"] == "500000"
    assert "--batch_size" not in flags
    assert "--seq_len" not in flags
    assert "--train_ratio" not in flags
