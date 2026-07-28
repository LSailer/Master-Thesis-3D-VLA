
"""SLURM launch-script regression tests.

Two kinds of test live here. Launcher *behavior* (extends resolution, mode
overrides, strict bash, setup hooks, the GPU-memory sidecar) is asserted against
configs discovered by globbing ``scripts/slurm/configs``, because the config set
turns over with every migrated experiment arm and a hardcoded list goes stale.
A handful of tests instead lock in one arm's validated run shape; those name
their config on purpose, and should be deleted with the arm.
"""
from __future__ import annotations

import contextlib
import importlib.util
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "scripts/slurm/configs"


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


def discovered_configs() -> list[str]:
    """Every launchable config name, found by globbing the config directory.

    ``_``-prefixed files are shared bases pulled in via ``extends``, not
    variants that can be submitted, so they are excluded.
    """
    return sorted(
        path.stem for path in CONFIG_DIR.glob("*.yaml") if not path.stem.startswith("_")
    )


CONFIGS = discovered_configs()


def _raw(name: str) -> dict[str, Any]:
    """The config's own YAML, before ``extends`` merging."""
    return yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text()) or {}


def _extends_chain(name: str) -> list[str]:
    """Config names from ``name`` up to its root base, inclusive."""
    chain = [name]
    while (parent := _raw(chain[-1]).get("extends")) is not None:
        assert parent not in chain, f"circular extends: {chain + [parent]}"
        chain.append(parent)
    return chain


def _first_declared(chain: list[str], *keys: str) -> Any:
    """Value of the nested ``keys`` from the nearest config in ``chain`` declaring it."""
    for name in chain:
        node: Any = _raw(name)
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        if node is not None:
            return node
    raise AssertionError(f"no config in {chain} declares {keys}")


def _first_launchable() -> str:
    """Any config that names a run; abstract ``extends`` parents null their run_id."""
    for name in CONFIGS:
        if launch.load_config(name).run_id is not None:
            return name
    raise AssertionError(f"no launchable config among {CONFIGS}")


# Stand-in for "some real config" in tests about launcher behavior rather than
# about one experiment arm.
ANY_CONFIG = _first_launchable()


def run_launch(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "scripts/slurm/launch.sh", *args],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
        env={**os.environ, **(env or {})},
    )


def training_command(text: str) -> tuple[str, str, str | None, dict[str, str | None]]:
    """Extract (python_cmd, script, run_id, {flag: value}) from a rendered
    sbatch's training invocation (the python line followed by `--flag value`
    continuations).

    ``run_id`` is the optional leading positional emitted for dispatcher
    entrypoints (``run.py``), else ``None``. The invocation head is the first
    unindented, non-comment line naming a ``.py`` script: the curriculum-check
    ``generate_curriculum.py`` guard sits inside an ``if`` block and is indented,
    and a config with no args renders its command without a trailing backslash.

    A boolean YAML arg renders as a bare ``--flag`` with no value token; those
    map to ``None`` rather than to the empty string, which stays reserved for an
    explicitly empty value (``--flag ""``)."""

    lines = text.splitlines()
    idx = next(
        i for i, line in enumerate(lines)
        if ".py" in line and not line.startswith(("#", " ", "\t"))
    )
    head = lines[idx].strip().rstrip(" \\")
    tokens = head.split()
    s = next(j for j, tok in enumerate(tokens) if tok.endswith(".py"))
    python_cmd = " ".join(tokens[:s])
    script = tokens[s]
    run_id = tokens[s + 1] if len(tokens) > s + 1 else None
    flags: dict[str, str | None] = {}
    for line in lines[idx + 1:]:
        stripped = line.strip().rstrip(" \\")
        if not stripped.startswith("--"):
            break
        flag, sep, value = stripped.partition(" ")
        flags[flag] = value.strip().strip('"') if sep else None
    return python_cmd, script, run_id, flags


# --------------------------------------------------------------------------- #
# Config discovery                                                             #
# --------------------------------------------------------------------------- #


def test_config_directory_is_not_empty() -> None:
    # Every discovery-driven test below silently becomes vacuous if the glob
    # stops matching, so assert the glob itself works.
    assert CONFIGS


@pytest.mark.parametrize("name", CONFIGS)
def test_every_config_resolves_and_renders_both_modes(name: str) -> None:
    """A config in the directory must validate and render, prod and smoke."""
    config = launch.load_config(name)

    for mode in ("prod", "smoke"):
        rendered = launch.render_sbatch(config, mode=mode)
        assert rendered.startswith("#!/bin/bash")
        assert f"#SBATCH --gres={config.sbatch.gres}" in rendered
        training_command(rendered)


# --------------------------------------------------------------------------- #
# s1 — preserved contract                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="BSD wc left-pads the fake sbatch job id; the launcher targets the Linux cluster",
)
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
        ANY_CONFIG,
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
# s2 — recursive `extends`                                                     #
# --------------------------------------------------------------------------- #


def test_extends_chain_is_recursive() -> None:
    """The deepest chain in the tree must fully resolve, leaf overrides winning.

    Picked by depth rather than by name so the assertion keeps exercising real
    multi-level inheritance as arms come and go.
    """
    chain = max((_extends_chain(name) for name in CONFIGS), key=len)
    assert len(chain) >= 3, f"no multi-level extends chain to exercise: {chain}"

    config = launch.load_config(chain[0])

    # The leaf's own keys win over every ancestor's.
    assert config.job_name == _raw(chain[0])["job_name"]
    # Keys only an ancestor declares are still inherited, however deep.
    assert config.script == _first_declared(chain, "script")
    assert config.sbatch.gres == _first_declared(chain, "sbatch", "gres")
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
        "#!/usr/bin/env bash\n"
        "cat >/dev/null\n"
        "printf 'x\\n' >> \"$SBATCH_CALLS\"\n"
        "wc -l < \"$SBATCH_CALLS\"\n"
    )
    fake_sbatch.chmod(0o755)
    variants = CONFIGS[:4]
    assert len(variants) == 4

    result = run_launch(
        *variants, "--smoke",
        env={"PATH": f"{tmp_path}:{os.environ['PATH']}", "SBATCH_CALLS": str(calls_file)},
    )

    assert result.returncode == 0, result.stderr
    assert len(calls_file.read_text().splitlines()) == len(variants)


# --------------------------------------------------------------------------- #
# Rendering behavior, asserted over whatever configs exist                     #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", CONFIGS)
def test_strict_bash_is_opt_in_for_prod_and_always_on_for_smoke(name: str) -> None:
    config = launch.load_config(name)

    prod = launch.render_sbatch(config, mode="prod")
    smoke = launch.render_sbatch(launch.load_config(name), mode="smoke")

    assert ("set -euo pipefail" in prod) is config.strict_bash
    assert "set -euo pipefail" in smoke


@pytest.mark.parametrize("name", CONFIGS)
def test_timestamp_is_defined_exactly_when_a_rendered_value_uses_it(name: str) -> None:
    # A bare ``${TIMESTAMP}`` in a path would expand to the empty string and
    # silently collapse every run into one directory.
    for mode in ("prod", "smoke"):
        config = launch.load_config(name)
        rendered = launch.render_sbatch(config, mode=mode)
        _sbatch, args = launch._resolve_mode(launch.load_config(name), mode)
        uses_timestamp = any(
            "${TIMESTAMP}" in str(value)
            for value in (*args.values(), *config.env.values())
        )
        assert ("TIMESTAMP=$(date +%Y%m%d-%H%M%S)" in rendered) is uses_timestamp


@pytest.mark.parametrize("name", CONFIGS)
def test_setup_hooks_are_rendered_before_the_training_command(name: str) -> None:
    config = launch.load_config(name)
    if not config.setup:
        pytest.skip(f"{name} declares no setup hooks")
    rendered = launch.render_sbatch(config, mode="smoke")

    training_line = rendered.index(config.script)
    for hook in config.setup:
        assert hook in rendered
        assert rendered.index(hook) < training_line


@pytest.mark.parametrize("name", CONFIGS)
def test_standalone_scripts_keep_their_own_interpreter_and_take_no_run_id(
    name: str,
) -> None:
    """Profiling-style configs run a script directly, with no run id and no --steps."""
    config = launch.load_config(name)
    if config.script.endswith("run.py"):
        pytest.skip(f"{name} uses the run.py dispatcher")
    rendered = launch.render_sbatch(config, mode="smoke")
    _python_cmd, script, run_id, flags = training_command(rendered)

    assert f"{config.python} {config.script}" in rendered
    assert script == config.script
    assert run_id is None
    assert "--steps" not in flags


# --------------------------------------------------------------------------- #
# Boolean rendering                                                            #
# --------------------------------------------------------------------------- #


# The one arm that carries a YAML boolean (`full_bf16: true`); if it is retired,
# point these at whatever config replaces it rather than deleting the contract.
BOOL_CONFIG = "hybrid_hpp_bf16_prodshape_probe"


@contextlib.contextmanager
def temp_config(name: str, body: str) -> Iterator[str]:
    """Write ``body`` as a throwaway config in the real config dir, yield its name."""
    path = CONFIG_DIR / f"{name}.yaml"
    path.write_text(body)
    try:
        yield name
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("mode", ["prod", "smoke"])
def test_true_boolean_renders_as_a_bare_flag(mode: str) -> None:
    """`full_bf16: true` must render `--full_bf16`, never `--full_bf16 True`."""
    config = launch.load_config(BOOL_CONFIG)
    assert config.args["full_bf16"] is True, "config no longer carries a YAML boolean"

    rendered = launch.render_sbatch(config, mode=mode)
    _python_cmd, _script, _run_id, flags = training_command(rendered)

    assert "--full_bf16" in flags
    assert flags["--full_bf16"] is None
    assert "--full_bf16 " not in rendered
    assert "True" not in flags.values()


def test_false_boolean_renders_no_line_at_all() -> None:
    body = (
        "extends: _base\n"
        "job_name: bool-false-probe\n"
        "output_dir: output/bool-false-probe\n"
        "args:\n"
        "  full_bf16: false\n"
    )
    with temp_config("bool_false_probe", body) as name:
        rendered = launch.render_sbatch(launch.load_config(name), mode="prod")

    _python_cmd, _script, _run_id, flags = training_command(rendered)
    assert "--full_bf16" not in flags
    assert "full_bf16" not in rendered
    # The remaining args still render, so the omission is surgical.
    assert flags["--steps"] == "2000000"


def test_quoted_boolean_string_is_rejected_by_name() -> None:
    """A YAML-quoted "true" is a string and would render as `--flag true`."""
    body = (
        "extends: _base\n"
        "job_name: bool-quoted-probe\n"
        "output_dir: output/bool-quoted-probe\n"
        'args:\n'
        '  full_bf16: "true"\n'
    )
    with (
        temp_config("bool_quoted_probe", body) as name,
        pytest.raises(ValueError, match="full_bf16") as excinfo,
    ):
        launch.load_config(name)

    assert "quoted string" in str(excinfo.value)


def test_quoted_boolean_under_smoke_args_is_rejected_too() -> None:
    """`smoke.args` is merged into the rendered command, so it validates alike."""
    body = (
        "extends: _base\n"
        "job_name: bool-smoke-quoted-probe\n"
        "output_dir: output/bool-smoke-quoted-probe\n"
        "smoke:\n"
        "  args:\n"
        '    full_bf16: "false"\n'
    )
    with (
        temp_config("bool_smoke_quoted_probe", body) as name,
        pytest.raises(ValueError, match="smoke.args.full_bf16") as excinfo,
    ):
        launch.load_config(name)

    assert "quoted string" in str(excinfo.value)


def test_quoted_false_string_is_rejected_too() -> None:
    # The dangerous half: `--flag false` renders as a pair whose meaning depends
    # on the target parser - a value for a value-taking one, a stray positional
    # for a store_true one - so it is rejected rather than guessed at.
    body = (
        "extends: _base\n"
        "job_name: bool-quoted-false-probe\n"
        "output_dir: output/bool-quoted-false-probe\n"
        'args:\n'
        '  full_bf16: "False"\n'
    )
    with (
        temp_config("bool_quoted_false_probe", body) as name,
        pytest.raises(ValueError, match="full_bf16"),
    ):
        launch.load_config(name)


def test_valueless_arg_is_rejected_as_a_non_scalar() -> None:
    """A bare `key:` loads as ``None``, which is not one of the four scalars.

    ``_validate_args`` promises its caller a ``dict[str, Scalar]``, so it admits
    exactly str/int/float/bool; a null would otherwise have rendered the literal
    ``--full_bf16 None``.
    """
    body = (
        "extends: _base\n"
        "job_name: null-arg-probe\n"
        "output_dir: output/null-arg-probe\n"
        "args:\n"
        "  full_bf16:\n"
    )
    with (
        temp_config("null_arg_probe", body) as name,
        pytest.raises(ValueError, match="full_bf16 must be a scalar") as excinfo,
    ):
        launch.load_config(name)

    assert "NoneType" in str(excinfo.value)


def test_bare_boolean_flag_still_parses_with_the_train_parser() -> None:
    """The rendered bf16 arm must survive argparse, bare flag included.

    The train parser takes ``--full_bf16`` with ``nargs="?"``/``const=True``, so
    the bare form is already accepted; this pins that the launcher and the CLI
    agree end to end rather than only on the option spelling.
    """
    from src.r2dreamer.launch.parser import _build_parser_train

    rendered = launch.render_sbatch(launch.load_config(BOOL_CONFIG), mode="prod")
    _python_cmd, _script, _run_id, flags = training_command(rendered)

    argv: list[str] = []
    for flag, value in flags.items():
        argv.append(flag)
        if value is not None:
            argv.append(value)

    args = _build_parser_train().parse_args(argv)
    assert args.full_bf16 is True


def _train_parser_option_strings() -> set[str]:
    from src.r2dreamer.launch.parser import build_parser_train

    return {
        option
        for action in build_parser_train()._actions
        for option in action.option_strings
    }


@pytest.mark.parametrize("name", CONFIGS)
def test_every_rendered_flag_is_accepted_by_the_train_parser(name: str) -> None:
    """A config may only render flags the train CLI still defines.

    The launcher passes ``args:`` through verbatim, so a knob deleted from the
    parser leaves a config that renders fine and then dies at ``argparse`` on the
    cluster. This is the only place that drift is visible without submitting.
    """
    config = launch.load_config(name)
    if not config.script.endswith("run.py"):
        pytest.skip(f"{name} runs a standalone script with its own flags")
    known = _train_parser_option_strings()

    for mode in ("prod", "smoke"):
        _sbatch, args = launch._resolve_mode(launch.load_config(name), mode)
        unknown = sorted(
            launch._flag(key, config.arg_style)
            for key in args
            if launch._flag(key, config.arg_style) not in known
        )
        assert not unknown, f"{name} ({mode}) renders unknown train flags: {unknown}"


def test_rendered_sbatch_logs_gpu_memory_csv() -> None:
    # Emitted for every job, so the log path is derived from the config rather
    # than pinned to one arm's output tree.
    config = launch.load_config(ANY_CONFIG)
    rendered = launch.render_sbatch(config, mode="prod")

    assert (
        f'GPU_MEMORY_LOG="{config.output_dir}/gpu-memory-${{SLURM_JOB_ID}}.csv"'
    ) in rendered
    assert "--query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu" in rendered
    assert '--format=csv -l 5 > "$GPU_MEMORY_LOG"' in rendered
    assert 'echo "GPU memory log: $GPU_MEMORY_LOG"' in rendered


def test_smoke_renders_a_pass_gate_when_the_config_asserts_an_artifact() -> None:
    # Inherited from _base: every curriculum smoke must prove it wrote metrics.
    config = launch.load_config(ANY_CONFIG)
    assert config.smoke.assert_file

    rendered = launch.render_sbatch(config, mode="smoke")

    assert config.smoke.assert_file in rendered
    assert "=== Smoke PASS ===" in rendered


# --------------------------------------------------------------------------- #
# Mode / CLI overrides                                                         #
# --------------------------------------------------------------------------- #


def test_time_override_applies_to_selected_mode() -> None:
    rendered = launch.render_sbatch(
        launch.load_config(ANY_CONFIG), mode="prod", time_override="04:00:00",
    )

    assert "#SBATCH --time=04:00:00" in rendered


def test_partition_override_applies_to_selected_mode() -> None:
    rendered = launch.render_sbatch(
        launch.load_config(ANY_CONFIG), mode="prod", partition_override="gpu_h100",
    )

    assert "#SBATCH --partition=gpu_h100" in rendered
    assert "#SBATCH --partition=gpu_h100_il,gpu_h100" not in rendered


def test_launch_wrapper_forwards_time_override() -> None:
    result = run_launch(ANY_CONFIG, "--prod", "--time", "04:00:00", "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "#SBATCH --time=04:00:00" in result.stdout


def test_launch_wrapper_forwards_partition_override() -> None:
    result = run_launch(ANY_CONFIG, "--prod", "--partition", "gpu_h100", "--dry-run")

    assert result.returncode == 0, result.stderr
    assert "#SBATCH --partition=gpu_h100" in result.stdout
    assert "#SBATCH --partition=gpu_h100_il,gpu_h100" not in result.stdout


def test_launch_wrapper_time_override_uses_default_prod_mode() -> None:
    config = launch.load_config(ANY_CONFIG)
    result = run_launch(ANY_CONFIG, "--time", "04:00:00", "--dry-run")

    assert result.returncode == 0, result.stderr
    assert f"#SBATCH --job-name={config.job_name}" in result.stdout
    assert "#SBATCH --time=04:00:00" in result.stdout
    assert f"--steps {config.args['steps']}" in result.stdout


# --------------------------------------------------------------------------- #
# Per-arm run shapes (delete a case together with its arm)                     #
# --------------------------------------------------------------------------- #


def test_full_tokens_smoke_run_shape() -> None:
    rendered = launch.render_sbatch(launch.load_config("full_tokens_l1"), mode="smoke")
    _, _, run_id, flags = training_command(rendered)

    assert run_id == "habitat-l1-full-tokens"
    assert "#SBATCH --job-name=smoke-rgb-full-tokens-l1" in rendered
    assert flags["--output_dir"] == "output/smoke/rgb-full-tokens-${TIMESTAMP}"
    assert flags["--wandb_name"] == "rgb-full-tokens-smoke-${TIMESTAMP}"
    assert "no-gate" in flags["--wandb_tags"]
    # A token row is 5.6 MB, so the capped replay capacity is load-bearing.
    assert flags["--buffer_capacity"] == "500"
    assert "=== Smoke PASS ===" in rendered


def test_global_tokens_smoke_run_shape() -> None:
    rendered = launch.render_sbatch(launch.load_config("global_tokens_l1"), mode="smoke")
    _, _, run_id, flags = training_command(rendered)

    assert run_id == "habitat-l1-vggt-house-global-tokens-nogate"
    assert "#SBATCH --job-name=smoke-rgb-global-tokens-l1" in rendered
    assert flags["--output_dir"] == "output/smoke/rgb-global-tokens-${TIMESTAMP}"
    assert flags["--buffer_capacity"] == "1000"
    assert "=== Smoke PASS ===" in rendered


def test_aggregator_pooled_prod_args() -> None:
    rendered = launch.render_sbatch(launch.load_config("aggregator_pooled_l1"), mode="prod")
    python_cmd, script, run_id, flags = training_command(rendered)

    assert python_cmd == "uv run python"
    assert script.endswith("run.py")
    assert run_id == "habitat-l1-aggregator-pooled"
    assert flags["--steps"] == "1500000"
    assert flags["--prefill"] == "5000"
    assert flags["--checkpoint_every"] == "100000"
    assert flags["--render_resolution"] == "518"
    assert "aggregator-pooled" in flags["--wandb_tags"]
    assert "skip-heads" in flags["--wandb_tags"]


def test_aggregator_pooled_smoke_overrides() -> None:
    rendered = launch.render_sbatch(launch.load_config("aggregator_pooled_l1"), mode="smoke")
    python_cmd, _, _, flags = training_command(rendered)

    # Smokes skip the slow uv dependency resync.
    assert python_cmd == "uv run --no-sync python"
    assert flags["--steps"] == "800"
    assert flags["--prefill"] == "200"
    assert "#SBATCH --time=00:20:00" in rendered
    assert "smoke" in flags["--wandb_tags"]
    # Timestamp-based run id requires a TIMESTAMP definition in the script body.
    assert "TIMESTAMP=$(date +%Y%m%d-%H%M%S)" in rendered
    assert flags["--output_dir"] == "output/smoke/aggregator-pooled-${TIMESTAMP}"


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


def test_gnn_smoke_config_renders_stability_gate() -> None:
    # Locks in the validated GNN smoke path (jobs 5744825/5744826): >=15-min
    # duration (4500 train steps, measured ~20 min on H100), the teardown
    # hard-exit guard, and the metrics gate.
    rendered = launch.render_sbatch(
        launch.load_config("gnn_house_points_pose_l1_live"), mode="smoke",
    )
    _, _, run_id, flags = training_command(rendered)

    assert run_id == "habitat-l1-gnn-house-points-pose"
    assert flags["--steps"] == "4500"
    assert flags["--prefill"] == "1000"
    assert "#SBATCH --partition=gpu_h100_short" in rendered
    assert 'export R2DREAMER_HARD_EXIT_ON_FINISH="1"' in rendered
    assert "metrics.csv" in rendered


def test_gnn_plydump_smoke_keeps_the_parent_shape_under_its_own_name() -> None:
    # A diagnostic leaf reuses the parent's validated smoke shape and only
    # re-points the job name and output tree.
    rendered = launch.render_sbatch(
        launch.load_config("gnn_house_points_pose_l1_live_plydump"), mode="smoke",
    )
    _, _, run_id, flags = training_command(rendered)

    assert run_id == "habitat-l1-gnn-house-points-pose"
    assert "#SBATCH --job-name=smoke-gnn-house-points-pose-l1-live-plydump" in rendered
    assert flags["--steps"] == "4500"
    assert "gnn-house-points-pose-live-plydump" in flags["--output_dir"]
    assert "=== Smoke PASS ===" in rendered


def test_hybrid_house_points_smoke_config_inherits_parent_shape() -> None:
    # The additive-hybrid variant reuses the parent's validated smoke shape
    # (1000 prefill + 2000 train) and maps to its own run_id/output tree.
    rendered = launch.render_sbatch(
        launch.load_config("hybrid_house_points_pose_l1_live"), mode="smoke"
    )
    _, _, run_id, flags = training_command(rendered)

    assert run_id == "habitat-l1-vggt-hybrid-house-points-pose"
    assert flags["--steps"] == "2000"
    assert flags["--prefill"] == "1000"
    assert "vggt-hybrid-house-points-pose-live" in flags["--output_dir"]


# --------------------------------------------------------------------------- #
# 3D-63 replay-capacity ablation / 3D-87 dtype probes                          #
# --------------------------------------------------------------------------- #


def _configs_tagged(fragment: str) -> list[str]:
    """Config names whose prod ``wandb_tags`` carry ``fragment`` as a whole tag."""
    tagged = []
    for name in CONFIGS:
        tags = str(launch.load_config(name).args.get("wandb_tags", ""))
        if fragment in tags.split(","):
            tagged.append(name)
    return tagged


REPLAY_CAPACITY_CONFIGS = _configs_tagged("replay-capacity")
DTYPE_PROBE_CONFIGS = _configs_tagged("dtype-plumbing-noop")


@pytest.mark.parametrize("variant", REPLAY_CAPACITY_CONFIGS)
def test_l1_replay_capacity_ablation_configs(variant: str) -> None:
    config = launch.load_config(variant)
    rendered = launch.render_sbatch(config, mode="prod")
    _, script, run_id, flags = training_command(rendered)

    assert script.endswith("run.py")
    assert run_id == config.run_id
    # The ablation varies capacity only: the seed and the scalars-only logging
    # shape are shared, or the success-rate comparison is not apples to apples.
    assert flags["--seed"] == "42"
    assert flags["--buffer_capacity"] == str(config.args["buffer_capacity"])
    assert "3d-63" in flags["--wandb_tags"]
    assert flags["--video_log_every"] == "0"
    assert flags["--val_every"] == "0"


def test_replay_capacity_ablation_covers_more_than_one_capacity() -> None:
    capacities = {
        launch.load_config(name).args["buffer_capacity"]
        for name in REPLAY_CAPACITY_CONFIGS
    }
    assert len(capacities) > 1, capacities


def test_capacity_override_does_not_drag_batch_defaults_along() -> None:
    rendered = launch.render_sbatch(launch.load_config("l1_cnn_cap10k"), mode="prod")
    _, _, _, flags = training_command(rendered)

    assert flags["--buffer_capacity"] == "10000"
    assert "--batch_size" not in flags
    assert "--seq_len" not in flags
    assert "--train_ratio" not in flags


@pytest.mark.parametrize("variant", DTYPE_PROBE_CONFIGS)
def test_3d87_probe_configs_render_expected_flags_paths_and_tags(variant: str) -> None:
    config = launch.load_config(variant)
    rendered = launch.render_sbatch(config, mode="prod")
    _, script, run_id, flags = training_command(rendered)

    assert script.endswith("run.py")
    assert run_id == config.run_id
    assert flags["--steps"] == "50000"
    assert flags["--prefill"] == "5000"
    assert flags["--checkpoint_every"] == "1000000"
    assert flags["--output_dir"] == config.args["output_dir"]
    assert flags["--seed"] == "42"
    assert flags["--wandb_name"] == config.args["wandb_name"]
    assert flags["--video_log_every"] == "0"
    assert flags["--val_every"] == "0"
    # The probe family exists to compare dtypes, so each arm must name one.
    assert flags["--compute_dtype"] == config.args["compute_dtype"]
    for fragment in ("3d-87", "seed-42"):
        assert fragment in flags["--wandb_tags"]


def test_3d87_probes_cover_more_than_one_compute_dtype() -> None:
    dtypes = {
        launch.load_config(name).args["compute_dtype"] for name in DTYPE_PROBE_CONFIGS
    }
    assert len(dtypes) > 1, dtypes
