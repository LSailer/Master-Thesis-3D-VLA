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
    # _base / l1_vggt have since gained four intentional changes the frozen
    # legacy script predates: (1) the GL-teardown hard-exit env var (so every
    # habitat variant exits 0 on completion), (2) multi-partition auto-select
    # (gpu_h100_il,gpu_h100), (3) the scalars-only flags video_log_every=0 /
    # val_every=0 (videos + eval regenerated from checkpoints, not during
    # training), and (4) GPU memory CSV logging. Normalise them back out before
    # the byte-equality check so the rest of the contract still holds.
    rendered = result.stdout.replace('export R2DREAMER_HARD_EXIT_ON_FINISH="1"\n\n', "", 1)
    rendered = rendered.replace(
        'GPU_MEMORY_LOG="output/r2dreamer-curriculum-l1-vggt/gpu-memory-${SLURM_JOB_ID}.csv"\n'
        'if command -v nvidia-smi >/dev/null 2>&1; then\n'
        '    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu --format=csv -l 5 > "$GPU_MEMORY_LOG" 2>/dev/null &\n'
        '    GPU_MONITOR_PID=$!\n'
        "    trap 'if [ -n \"${GPU_MONITOR_PID:-}\" ]; then kill \"$GPU_MONITOR_PID\" 2>/dev/null || true; wait \"$GPU_MONITOR_PID\" 2>/dev/null || true; fi' EXIT\n"
        '    echo "GPU memory log: $GPU_MEMORY_LOG"\n'
        'fi\n\n',
        "\n",
        1,
    )
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


def test_time_override_applies_to_selected_mode() -> None:
    rendered = launch.render_sbatch(
        launch.load_config("house_context_l1"), mode="prod", time_override="04:00:00",
    )

    assert "#SBATCH --time=04:00:00" in rendered
    assert "habitat-l1-vggt-house-context" in rendered


def test_partition_override_applies_to_selected_mode() -> None:
    rendered = launch.render_sbatch(
        launch.load_config("house_context_l1"), mode="prod", partition_override="gpu_h100",
    )

    assert "#SBATCH --partition=gpu_h100" in rendered
    assert "#SBATCH --partition=gpu_h100_il,gpu_h100" not in rendered


def test_launch_wrapper_forwards_time_override() -> None:
    result = run_launch("house_context_l1", "--prod", "--time", "04:00:00", "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "#SBATCH --time=04:00:00" in result.stdout


def test_launch_wrapper_forwards_partition_override() -> None:
    result = run_launch("house_context_l1", "--prod", "--partition", "gpu_h100", "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "#SBATCH --partition=gpu_h100" in result.stdout
    assert "#SBATCH --partition=gpu_h100_il,gpu_h100" not in result.stdout


def test_launch_wrapper_time_override_uses_default_prod_mode() -> None:
    result = run_launch("house_context_l1", "--time", "04:00:00", "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "#SBATCH --job-name=vggt-house-context-l1" in result.stdout
    assert "#SBATCH --time=04:00:00" in result.stdout
    assert "--steps 2000000" in result.stdout


def test_house_full_tokens_nogate_smoke_uses_new_run_id() -> None:
    rendered = launch.render_sbatch(
        launch.load_config("house_full_tokens_nogate_l1"), mode="smoke",
    )
    _, _, run_id, flags = training_command(rendered)

    assert run_id == "habitat-l1-vggt-house-full-tokens-nogate"
    assert "#SBATCH --job-name=smoke-vggt-house-full-tokens-nogate-l1" in rendered
    assert flags["--output_dir"] == "output/smoke/vggt-house-full-tokens-nogate-${TIMESTAMP}"
    assert flags["--wandb_name"] == "vggt-house-full-tokens-nogate-smoke-${TIMESTAMP}"
    assert "no-gate" in flags["--wandb_tags"]
    assert "=== Smoke PASS ===" in rendered


def test_house_global_embedding_smoke_uses_new_run_id_and_dump_knob() -> None:
    rendered = launch.render_sbatch(
        launch.load_config("house_global_embedding_l1"), mode="smoke",
    )
    _, _, run_id, flags = training_command(rendered)

    assert run_id == "habitat-l1-vggt-house-global-embedding"
    assert "#SBATCH --job-name=smoke-vggt-house-global-embedding-l1" in rendered
    assert flags["--output_dir"] == "output/smoke/vggt-house-global-embedding-${TIMESTAMP}"
    assert flags["--pointcloud_dump_every"] == "300"
    assert "pointnet-reducer" in flags["--wandb_tags"]
    assert "=== Smoke PASS ===" in rendered


def test_house_context_long_smoke_runs_past_warmup_with_buffer() -> None:
    rendered = launch.render_sbatch(
        launch.load_config("house_context_l1_long_smoke"), mode="smoke",
    )
    _, _, run_id, flags = training_command(rendered)

    assert run_id == "habitat-l1-vggt-house-context"
    assert "#SBATCH --job-name=smoke-vggt-house-context-l1-long-smoke" in rendered
    assert "#SBATCH --partition=gpu_h100_short" in rendered
    assert "#SBATCH --time=00:30:00" in rendered
    assert flags["--steps"] == "12000"
    assert flags["--prefill"] == "200"
    assert flags["--batch_size"] == "4"
    assert flags["--seq_len"] == "16"
    assert flags["--train_ratio"] == "16"
    assert flags["--log_every"] == "500"
    assert flags["--checkpoint_every"] == "100000"
    assert flags["--output_dir"] == "output/smoke-long/vggt-house-context-${TIMESTAMP}"
    assert flags["--wandb_name"] == "vggt-house-context-long-smoke-${TIMESTAMP}"
    assert "long-smoke" in flags["--wandb_tags"]
    assert "=== Smoke PASS ===" in rendered


@pytest.mark.parametrize(
    "variant,script",
    [
        ("profile_encoder_cost", "scripts/profiling/profile_encoders_3d5253.py"),
        ("profile_training_vggt", "scripts/profiling/profile_training.py"),
        ("profile_agg_pipeline", "scripts/profiling/profile_pipeline_aggregator_mlp.py"),
    ],
)
def test_profiling_configs_render_standalone_scripts(variant: str, script: str) -> None:
    rendered = launch.render_sbatch(launch.load_config(variant), mode="smoke")

    assert f".venv/bin/python {script}" in rendered
    assert "output/profiling" in rendered
    assert "scripts/slurm/hooks/link_external.sh" in rendered
    assert "--steps" not in rendered

def test_aggregator_prod_is_strict_bash() -> None:
    rendered = launch.render_sbatch(launch.load_config("aggregator_mlp_v1"), mode="prod")
    assert "set -euo pipefail" in rendered


@pytest.mark.parametrize(
    "variant,run_id,capacity,tag",
    [
        ("l1_cnn_cap1m", "habitat-l1-cnn", "1000000", "cap-1m"),
        ("l1_cnn_cap500k", "habitat-l1-cnn", "500000", "cap-500k"),
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


@pytest.mark.parametrize(
    "variant",
    [
        "l1_vggt_wpcp37_cap500k",
        "l1_vggt_wpcp64_cap500k",
        "l1_agg_mlp_cap500k",
    ],
)
def test_l1_vggt_capacity_ablation_links_external_repos(variant: str) -> None:
    rendered = launch.render_sbatch(launch.load_config(variant), mode="smoke")

    assert "./scripts/slurm/hooks/link_external.sh" in rendered


def test_rendered_sbatch_logs_gpu_memory_csv() -> None:
    rendered = launch.render_sbatch(
        launch.load_config("l1_cnn_cap1m_seed42_fp32_probe"), mode="prod",
    )

    assert (
        'GPU_MEMORY_LOG="output/probes/3d-87/cnn-cap1m-fp32-seed42/'
        'gpu-memory-${SLURM_JOB_ID}.csv"'
    ) in rendered
    assert "--query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu" in rendered
    assert '--format=csv -l 5 > "$GPU_MEMORY_LOG"' in rendered
    assert 'echo "GPU memory log: $GPU_MEMORY_LOG"' in rendered


@pytest.mark.parametrize(
    "variant,dtype_flag,run_id,output_dir,wandb_name,tag_fragments,steps",
    [
        (
            "house_full_tokens_nogate_prodshape_probe_l1",
            "float32",
            "habitat-l1-vggt-house-full-tokens-nogate",
            "output/probes/3d-87/full-token-nogate-fp32-prodshape/run-${SLURM_JOB_ID}",
            "full-token-nogate-fp32-prodshape-probe-${SLURM_JOB_ID}",
            [
                "3d-87", "prodshape-probe", "fp32", "full-token-transformer",
                "no-gate", "b16", "t64", "seed-42",
            ],
            "6000",
        ),
        (
            "house_full_tokens_nogate_bf16_prodshape_probe_l1",
            "bfloat16",
            "habitat-l1-vggt-house-full-tokens-nogate",
            "output/probes/3d-87/full-token-nogate-bf16-prodshape/run-${SLURM_JOB_ID}",
            "full-token-nogate-bf16-prodshape-probe-${SLURM_JOB_ID}",
            [
                "3d-87", "prodshape-probe", "bf16", "full-token-transformer",
                "no-gate", "b16", "t64", "seed-42",
            ],
            "6000",
        ),
        (
            "l1_cnn_cap1m_seed42_fp32_probe",
            "float32",
            "habitat-l1-cnn",
            "output/probes/3d-87/cnn-cap1m-fp32-seed42/run-${SLURM_JOB_ID}",
            "l1-cnn-cap1m-fp32-seed42-probe-${SLURM_JOB_ID}",
            ["3d-87", "cnn-baseline", "dtype-plumbing-noop", "fp32", "seed-42", "cap-1m"],
            "50000",
        ),
        (
            "l1_cnn_cap1m_seed42_bf16_probe",
            "bfloat16",
            "habitat-l1-cnn",
            "output/probes/3d-87/cnn-cap1m-bf16-seed42/run-${SLURM_JOB_ID}",
            "l1-cnn-cap1m-bf16-seed42-probe-${SLURM_JOB_ID}",
            ["3d-87", "cnn-baseline", "dtype-plumbing-noop", "bf16", "seed-42", "cap-1m"],
            "50000",
        ),
    ],
)
def test_3d87_probe_configs_render_expected_flags_paths_and_tags(
    variant: str,
    dtype_flag: str | None,
    run_id: str,
    output_dir: str,
    wandb_name: str,
    tag_fragments: list[str],
    steps: str,
) -> None:
    rendered = launch.render_sbatch(launch.load_config(variant), mode="prod")
    _, script, rendered_run_id, flags = training_command(rendered)

    assert script.endswith("run.py")
    assert rendered_run_id == run_id
    assert flags["--steps"] == steps
    assert flags["--prefill"] == "5000"
    assert flags["--checkpoint_every"] == "1000000"
    assert flags["--output_dir"] == output_dir
    assert flags["--seed"] == "42"
    if variant.startswith("house_full_tokens_nogate_"):
        assert flags["--batch_size"] == "16"
        assert flags["--seq_len"] == "64"
    assert flags["--wandb_name"] == wandb_name
    assert flags["--video_log_every"] == "0"
    assert flags["--val_every"] == "0"
    if dtype_flag is None:
        assert "--compute_dtype" not in flags
    else:
        assert flags["--compute_dtype"] == dtype_flag
    for fragment in tag_fragments:
        assert fragment in flags["--wandb_tags"]


@pytest.mark.parametrize(
    "variant, run_id",
    [
        ("gnn_house_points_pose_l1_live", "habitat-l1-gnn-house-points-pose"),
        ("gnn_edge_house_points_pose_l1_live", "habitat-l1-gnn-edge-house-points-pose"),
    ],
)
def test_gnn_smoke_configs_render_stability_gate(variant: str, run_id: str) -> None:
    # Locks in the validated GNN smoke path (jobs 5744825/5744826): >=15-min
    # duration (4500 train steps, measured ~20 min on H100), the teardown
    # hard-exit guard, and the metrics gate.
    rendered = launch.render_sbatch(launch.load_config(variant), mode="smoke")
    _, _, positional, flags = training_command(rendered)

    assert positional == run_id
    assert flags["--steps"] == "4500"
    assert flags["--prefill"] == "1000"
    assert "#SBATCH --partition=gpu_h100_short" in rendered
    assert 'export R2DREAMER_HARD_EXIT_ON_FINISH="1"' in rendered
    assert "metrics.csv" in rendered
