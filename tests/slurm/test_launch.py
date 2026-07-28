
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
import re
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


def _first_full_shape() -> str:
    """The first config carrying both a step budget and a smoke artifact gate.

    Several launcher-behavior tests read ``args["steps"]`` or the smoke pass
    gate off whichever config they are handed, so the stand-in has to be one
    that declares them rather than simply the alphabetically first.
    """
    for name in CONFIGS:
        config = launch.load_config(name)
        if "steps" in config.args and config.smoke.assert_file:
            return name
    raise AssertionError(f"no fully-shaped config among {CONFIGS}")


# Stand-in for "some real config" in tests about launcher behavior rather than
# about one experiment arm.
ANY_CONFIG = _first_full_shape()


def run_launch(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "scripts/slurm/launch.sh", *args],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
        env={**os.environ, **(env or {})},
    )


# The training invocation head: an unindented interpreter prefix followed by the
# ``-m <module>`` entrypoint, with or without a line continuation. Anchored at
# column zero on purpose - the curriculum-check guard runs
# ``generate_curriculum.py`` from inside an indented ``if`` block, and a config
# with no args renders its command without a trailing backslash.
COMMAND_HEAD = re.compile(r"^(?P<python>\S.*python) (?P<script>-m \S+)(?: \\)?$")


def training_command(text: str) -> tuple[str, str, dict[str, str | None]]:
    """Extract (python_cmd, script, {flag: value}) from a rendered sbatch's
    training invocation (the python line followed by `--flag value`
    continuations).

    A boolean YAML arg renders as a bare ``--flag`` with no value token; those
    map to ``None`` rather than to the empty string, which stays reserved for an
    explicitly empty value (``--flag ""``)."""

    lines = text.splitlines()
    idx, head = next(
        (i, match)
        for i, line in enumerate(lines)
        if (match := COMMAND_HEAD.match(line))
    )
    flags: dict[str, str | None] = {}
    for line in lines[idx + 1:]:
        stripped = line.strip().rstrip(" \\")
        if not stripped.startswith("--"):
            break
        flag, sep, value = stripped.partition(" ")
        flags[flag] = value.strip().strip('"') if sep else None
    return head["python"], head["script"], flags


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
# s1 - preserved contract                                                      #
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

    (``script`` is supplied by ``_base`` - the shared ``-m src.main`` entry
    point - so ``job_name`` is the structural field a leaf can still omit.)"""

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
# s2 - recursive `extends`                                                     #
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
def test_strict_bash_is_smoke_only(name: str) -> None:
    """Smokes abort on the first failing line; prod runs do not.

    A prod job that loses one setup hook should still reach the training
    command and fail there, where the run dir and the logs say why.
    """
    prod = launch.render_sbatch(launch.load_config(name), mode="prod")
    smoke = launch.render_sbatch(launch.load_config(name), mode="smoke")

    assert "set -euo pipefail" not in prod
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
def test_every_config_launches_the_single_entry_point(name: str) -> None:
    """No config carries a dispatcher positional any more: the arm is a flag.

    A leftover positional would be read by ``src.main``'s parser as an unknown
    argument and kill the job at startup, after the node was allocated.
    """
    config = launch.load_config(name)
    assert config.script == "-m src.main"

    for mode in ("prod", "smoke"):
        rendered = launch.render_sbatch(launch.load_config(name), mode=mode)
        _python_cmd, script, flags = training_command(rendered)

        assert script == "-m src.main"
        assert flags["--env"] == "habitat"
        assert flags["--adapter"]
        assert flags["--curriculum"]


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
    _python_cmd, _script, flags = training_command(rendered)

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

    _python_cmd, _script, flags = training_command(rendered)
    assert "--full_bf16" not in flags
    assert "full_bf16" not in rendered
    # The remaining args still render, so the omission is surgical.
    assert flags["--steps"] == "2000000"


# Every spelling a reader takes for a boolean but the loader hands back as a
# string: the quoted YAML 1.2 booleans, plus the YAML 1.1 words that ruamel's
# safe loader no longer resolves. Mixed case included, since the guard folds it.
BOOLEAN_LOOKING_STRINGS = [
    '"true"', '"false"', '"False"', "yes", "no", "on", "off", "Yes", "OFF",
]


@pytest.mark.parametrize("literal", BOOLEAN_LOOKING_STRINGS)
def test_boolean_looking_string_is_rejected_by_name(literal: str) -> None:
    """A string that reads as a boolean would render as a `--flag <word>` pair.

    What that pair means depends on the target parser - a value for a
    value-taking flag, a stray positional for a ``store_true`` one - so the
    launcher refuses it instead of guessing, naming the offending key.
    """
    body = (
        "extends: _base\n"
        "job_name: bool-string-probe\n"
        "output_dir: output/bool-string-probe\n"
        "args:\n"
        f"  full_bf16: {literal}\n"
    )
    with (
        temp_config("bool_string_probe", body) as name,
        pytest.raises(ValueError, match="args.full_bf16") as excinfo,
    ):
        launch.load_config(name)

    assert "not a boolean" in str(excinfo.value)


@pytest.mark.parametrize("literal", BOOLEAN_LOOKING_STRINGS)
def test_boolean_looking_string_under_smoke_args_is_rejected_too(literal: str) -> None:
    """`smoke.args` is merged into the rendered command, so it validates alike."""
    body = (
        "extends: _base\n"
        "job_name: bool-smoke-string-probe\n"
        "output_dir: output/bool-smoke-string-probe\n"
        "smoke:\n"
        "  args:\n"
        f"    full_bf16: {literal}\n"
    )
    with (
        temp_config("bool_smoke_string_probe", body) as name,
        pytest.raises(ValueError, match="smoke.args.full_bf16") as excinfo,
    ):
        launch.load_config(name)

    assert "not a boolean" in str(excinfo.value)


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


# A rendered value keeps its shell placeholders (`${SEED}`, `${SLURM_JOB_ID}`,
# `${TIMESTAMP}`) because bash expands them on the node. argparse does not, and
# `--seed ${SEED}` is not an int, so the drift test substitutes a value that is
# valid for every flag type the launcher can render.
SHELL_PLACEHOLDER = re.compile(r"\$\{[^}]+\}")


def rendered_argv(text: str) -> list[str]:
    """The training command's flags as an argv list argparse can consume."""
    _python_cmd, _script, flags = training_command(text)
    argv: list[str] = []
    for flag, value in flags.items():
        argv.append(flag)
        if value is not None:
            argv.append(SHELL_PLACEHOLDER.sub("1", value))
    return argv


def test_bare_boolean_flag_still_parses_with_the_launch_parser() -> None:
    """The rendered bf16 arm must survive argparse, bare flag included.

    ``--full_bf16`` is a ``store_true``, so the bare flag the launcher renders
    for ``full_bf16: true`` is the only form it accepts; this pins that the
    launcher and the CLI agree end to end rather than only on the spelling.
    """
    from src.launch.parser import build_parser

    rendered = launch.render_sbatch(launch.load_config(BOOL_CONFIG), mode="prod")

    args = build_parser().parse_args(rendered_argv(rendered))
    assert args.full_bf16 is True


@pytest.mark.parametrize("name", CONFIGS)
def test_every_rendered_command_parses_with_the_launch_parser(name: str) -> None:
    """Every config, in both modes, must render a command ``src.main`` accepts.

    The launcher passes ``args:`` through verbatim, so a knob renamed or deleted
    in the parser leaves a config that renders fine and then dies at argparse on
    the cluster, after the node was allocated. Parsing the rendered argv (rather
    than comparing option-string sets) also catches a wrong value type or an
    out-of-range choice, which is the other half of the same drift.
    """
    from src.launch.parser import build_parser

    for mode in ("prod", "smoke"):
        rendered = launch.render_sbatch(launch.load_config(name), mode=mode)
        argv = rendered_argv(rendered)
        try:
            parsed = build_parser().parse_args(argv)
        except SystemExit as exc:  # argparse exits on an unknown or bad flag
            raise AssertionError(
                f"{name} ({mode}) renders a command src.main rejects: {argv}"
            ) from exc

        assert parsed.mode == "train"  # no config overrides the parser default


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
    _, _, flags = training_command(rendered)

    assert flags["--adapter"] == "rgb_full_tokens"
    assert flags["--curriculum"] == "L1"
    assert "#SBATCH --job-name=smoke-rgb-full-tokens-l1" in rendered
    assert flags["--output_dir"] == "output/smoke/rgb-full-tokens-${TIMESTAMP}"
    assert flags["--wandb_name"] == "rgb-full-tokens-smoke-${TIMESTAMP}"
    assert "no-gate" in flags["--wandb_tags"]
    # A token row is 5.6 MB, so the capped replay capacity is load-bearing.
    assert flags["--buffer_capacity"] == "500"
    assert "=== Smoke PASS ===" in rendered


def test_global_tokens_smoke_run_shape() -> None:
    rendered = launch.render_sbatch(launch.load_config("global_tokens_l1"), mode="smoke")
    _, _, flags = training_command(rendered)

    assert flags["--adapter"] == "rgb_global_tokens"
    assert flags["--curriculum"] == "L1"
    assert "#SBATCH --job-name=smoke-rgb-global-tokens-l1" in rendered
    assert flags["--output_dir"] == "output/smoke/rgb-global-tokens-${TIMESTAMP}"
    assert flags["--buffer_capacity"] == "1000"
    assert "=== Smoke PASS ===" in rendered


def test_aggregator_pooled_prod_args() -> None:
    rendered = launch.render_sbatch(launch.load_config("aggregator_pooled_l1"), mode="prod")
    python_cmd, script, flags = training_command(rendered)

    assert python_cmd == "uv run python"
    assert script == "-m src.main"
    assert flags["--adapter"] == "aggregator_pooled"
    assert flags["--curriculum"] == "L1"
    assert flags["--steps"] == "1500000"
    assert flags["--prefill"] == "5000"
    assert flags["--checkpoint_every"] == "100000"
    assert flags["--render_resolution"] == "518"
    assert "aggregator-pooled" in flags["--wandb_tags"]
    assert "skip-heads" in flags["--wandb_tags"]


def test_aggregator_pooled_smoke_overrides() -> None:
    rendered = launch.render_sbatch(launch.load_config("aggregator_pooled_l1"), mode="smoke")
    python_cmd, _, flags = training_command(rendered)

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
    _, _, flags = training_command(rendered)

    assert flags["--adapter"] == "rgb_house_cloud_episodes"
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
    _, _, flags = training_command(rendered)

    assert flags["--adapter"] == "rgb_house_voxels_gnn"
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
    _, _, flags = training_command(rendered)

    assert flags["--adapter"] == "rgb_house_voxels_gnn"
    assert "#SBATCH --job-name=smoke-gnn-house-points-pose-l1-live-plydump" in rendered
    assert flags["--steps"] == "4500"
    assert "gnn-house-points-pose-live-plydump" in flags["--output_dir"]
    assert "=== Smoke PASS ===" in rendered


def test_hybrid_house_points_smoke_config_inherits_parent_shape() -> None:
    # The additive-hybrid variant reuses the parent's validated smoke shape
    # (1000 prefill + 2000 train) and names its own adapter and output tree.
    rendered = launch.render_sbatch(
        launch.load_config("hybrid_house_points_pose_l1_live"), mode="smoke"
    )
    _, _, flags = training_command(rendered)

    assert flags["--adapter"] == "rgb_house_voxels"
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
    _, script, flags = training_command(rendered)

    assert script == "-m src.main"
    assert flags["--adapter"] == config.args["adapter"]
    # The ablation varies capacity only: the seed and the scalars-only logging
    # shape are shared, or the success-rate comparison is not apples to apples.
    assert flags["--seed"] == "42"
    assert flags["--buffer_capacity"] == str(config.args["buffer_capacity"])
    assert "3d-63" in flags["--wandb_tags"]
    assert flags["--video_log_every"] == "0"


def test_replay_capacity_ablation_covers_more_than_one_capacity() -> None:
    capacities = {
        launch.load_config(name).args["buffer_capacity"]
        for name in REPLAY_CAPACITY_CONFIGS
    }
    assert len(capacities) > 1, capacities


def test_capacity_override_does_not_drag_batch_defaults_along() -> None:
    rendered = launch.render_sbatch(launch.load_config("l1_cnn_cap10k"), mode="prod")
    _, _, flags = training_command(rendered)

    assert flags["--buffer_capacity"] == "10000"
    assert "--batch_size" not in flags
    assert "--seq_len" not in flags
    assert "--train_ratio" not in flags


@pytest.mark.parametrize("variant", DTYPE_PROBE_CONFIGS)
def test_3d87_probe_configs_render_expected_flags_paths_and_tags(variant: str) -> None:
    config = launch.load_config(variant)
    rendered = launch.render_sbatch(config, mode="prod")
    _, script, flags = training_command(rendered)

    assert script == "-m src.main"
    assert flags["--adapter"] == config.args["adapter"]
    assert flags["--steps"] == "50000"
    assert flags["--prefill"] == "5000"
    assert flags["--checkpoint_every"] == "1000000"
    assert flags["--output_dir"] == config.args["output_dir"]
    assert flags["--seed"] == "42"
    assert flags["--wandb_name"] == config.args["wandb_name"]
    assert flags["--video_log_every"] == "0"
    # The probe family exists to compare dtypes, so each arm must name one.
    assert flags["--compute_dtype"] == config.args["compute_dtype"]
    for fragment in ("3d-87", "seed-42"):
        assert fragment in flags["--wandb_tags"]


def test_3d87_probes_cover_more_than_one_compute_dtype() -> None:
    dtypes = {
        launch.load_config(name).args["compute_dtype"] for name in DTYPE_PROBE_CONFIGS
    }
    assert len(dtypes) > 1, dtypes
