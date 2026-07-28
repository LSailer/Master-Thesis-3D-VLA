#!/usr/bin/env python3
"""Render and validate YAML-backed Slurm launch configs.

One universal renderer turns a per-variant YAML config (optionally extending a
base via ``extends:``) into a complete sbatch script. ``launch.sh`` pipes the
rendered script to ``sbatch``. Validation is fail-fast: a malformed config
raises *before* any job is submitted.

The schema is intentionally job-family-agnostic. A config supplies a ``script``
entrypoint plus a free-form ``args`` mapping; the renderer turns each ``args``
entry into a ``--flag value`` line (hyphen- or underscore-styled, auto-quoted),
except booleans, which render as a bare ``--flag`` (true) or not at all (false).
Optional ``env``/``setup``/``curriculum_check`` blocks cover env vars, pre-run
hooks, and the curriculum-generation guard used by the habitat training jobs.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from ruamel.yaml import YAML


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "scripts" / "slurm" / "configs"

Mode = Literal["prod", "smoke"]


class SbatchConfig(BaseModel):
    """#SBATCH resource directives shared by all modes."""

    model_config = ConfigDict(extra="forbid")

    partition: str
    gres: str
    ntasks: int
    cpus_per_task: int
    mem: str
    time: str


def _validate_args(value: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    """Reject args entries the renderer cannot turn into a well-formed flag.

    Applied to every mapping that feeds the rendered command - both the
    top-level ``args`` and ``smoke.args``, which is merged onto it in smoke
    mode - so the same config is accepted or rejected regardless of the mode
    it would be submitted in.
    """

    for key, item in value.items():
        if isinstance(item, (dict, list)):
            raise ValueError(
                f"{prefix}.{key} must be a scalar (str/int/float/bool), got {type(item).__name__}"
            )
        # A quoted boolean is a string, so it would render as the pair
        # `--flag false` instead of the bare switch a real bool renders as,
        # and what that pair means depends on how the target parser declares
        # the flag. Reject it outright and ask for the unquoted YAML boolean.
        if isinstance(item, str) and item.lower() in ("true", "false"):
            raise ValueError(
                f"{prefix}.{key} is the quoted string {item!r}, not a boolean; "
                f"write `{key}: {item.lower()}` unquoted so it renders as a flag"
            )
    return value


class SmokeConfig(BaseModel):
    """Mode-specific overrides for short dev-cluster smoke submissions."""

    model_config = ConfigDict(extra="forbid")

    partition: str = "gpu_h100_short"
    time: str = "00:30:00"
    args: dict[str, Any] = Field(default_factory=dict)
    assert_file: str | None = None  # path under the run dir the smoke must produce
    assert_min_rows: int | None = None  # minimum `wc -l` for assert_file (if a table)

    @field_validator("args")
    @classmethod
    def _args_scalar(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_args(value, prefix="smoke.args")


class LaunchConfig(BaseModel):
    """Fully merged + validated launch config for one variant."""

    model_config = ConfigDict(extra="forbid")

    job_name: str
    output_dir: str  # directory for #SBATCH logs (the run dir lives in args)
    script: str  # repo-relative python entrypoint
    sbatch: SbatchConfig

    # Optional leading positional for a dispatcher entrypoint (e.g. run.py's
    # run id). Rendered as ``<python> <script> <run_id> <flags>``; when unset
    # the command is just ``<python> <script> <flags>`` as before.
    run_id: str | None = None
    python: str = "uv run python"  # interpreter prefix (e.g. ".venv/bin/python")
    arg_style: Literal["underscore", "hyphen"] = "underscore"
    strict_bash: bool = False  # emit `set -euo pipefail` in prod too (always on for smoke)
    curriculum_check: str | None = None  # json path; emits a generate-if-missing guard

    args: dict[str, Any] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    setup: list[str] = Field(default_factory=list)  # raw bash lines run before training
    comments: list[str] = Field(default_factory=list)
    smoke: SmokeConfig = Field(default_factory=SmokeConfig)

    @field_validator("script")
    @classmethod
    def _script_relative(cls, value: str) -> str:
        if Path(value).is_absolute():
            raise ValueError("script must be repo-relative")
        return value

    @field_validator("args")
    @classmethod
    def _args_scalar(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_args(value, prefix="args")


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` onto ``base`` (overlay wins; lists replace)."""

    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    data = YAML(typ="safe").load(path)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def _config_path(name: str) -> Path:
    if "/" in name or name.startswith("."):
        raise ValueError(f"Invalid variant name: {name!r}")
    if not name.endswith(".yaml"):
        name = f"{name}.yaml"
    return CONFIG_DIR / name


def _resolve_raw(name: str, _seen: tuple[str, ...] = ()) -> dict[str, Any]:
    """Read a config and recursively merge its full ``extends:`` chain."""

    if name in _seen:
        chain = " -> ".join((*_seen, name))
        raise ValueError(f"Circular extends chain: {chain}")
    raw = _read_yaml(_config_path(name))
    parent = raw.pop("extends", None)
    if parent:
        parent_raw = _resolve_raw(str(parent), (*_seen, name))
        raw = _deep_merge(parent_raw, raw)
    return raw


def load_config(name: str, *, env_overrides: dict[str, str] | None = None) -> LaunchConfig:
    """Load, merge, and validate a launch config by variant name."""

    raw = _resolve_raw(name)
    if env_overrides:
        merged_env = dict(raw.get("env") or {})
        merged_env.update(env_overrides)
        raw["env"] = merged_env

    try:
        return LaunchConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid Slurm config for {name!r}:\n{exc}") from exc


def _resolve_mode(config: LaunchConfig, mode: Mode) -> tuple[SbatchConfig, dict[str, Any]]:
    sbatch = config.sbatch.model_copy(deep=True)
    args = dict(config.args)
    if mode == "smoke":
        sbatch.partition = config.smoke.partition
        sbatch.time = config.smoke.time
        args.update(config.smoke.args)  # in-place for existing keys, preserving order
    return sbatch, args


def _flag(name: str, arg_style: str) -> str:
    if arg_style == "hyphen":
        name = name.replace("_", "-")
    return f"--{name}"


def _needs_quote(value: str) -> bool:
    """Quote shell-significant values (matches the hand-written sbatch convention)."""

    return value == "" or any(ch in value for ch in " \t$,")


def _format_arg(name: str, value: Any, arg_style: str) -> str | None:
    """Render one ``args`` entry as an indented command-line continuation.

    Booleans are rendered as switches rather than as ``--flag value`` pairs:
    ``True`` emits the bare flag, ``False`` emits nothing at all. Every other
    value keeps the ``--flag value`` form, quoted when shell-significant.

    Args:
      name: The YAML ``args`` key.
      value: The YAML value; ``bool`` is special-cased, anything else is
        stringified.
      arg_style: ``"underscore"`` or ``"hyphen"`` flag spelling.

    Returns:
      The rendered line, or ``None`` when the entry contributes no line
      (a boolean that is ``False``).
    """

    flag = _flag(name, arg_style)
    if isinstance(value, bool):
        return f"    {flag}" if value else None
    rendered = str(value)
    if _needs_quote(rendered):
        rendered = f'"{rendered}"'
    return f"    {flag} {rendered}"


def _python_cmd(config: LaunchConfig, mode: Mode) -> str:
    # Skip the (slow) dependency resync for the default uv interpreter on smokes.
    if mode == "smoke" and config.python == "uv run python":
        return "uv run --no-sync python"
    return config.python


def _run_dir(args: dict[str, Any], config: LaunchConfig) -> str:
    return str(args.get("output_dir") or args.get("out_dir") or config.output_dir)


def render_sbatch(
    config: LaunchConfig, *,
    mode: Mode = "prod",
    time_override: str | None = None,
    partition_override: str | None = None,
) -> str:
    """Render a complete sbatch script for ``mode`` ("prod" or "smoke")."""

    sbatch, args = _resolve_mode(config, mode)
    if time_override is not None:
        sbatch.time = time_override
    if partition_override is not None:
        if not partition_override:
            raise ValueError("partition override must not be empty")
        sbatch.partition = partition_override
    log_dir = config.output_dir if mode == "prod" else f"{config.output_dir}/smoke"
    job_name = config.job_name if mode == "prod" else f"smoke-{config.job_name}"

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={sbatch.partition}",
        f"#SBATCH --gres={sbatch.gres}",
        f"#SBATCH --ntasks={sbatch.ntasks}",
        f"#SBATCH --cpus-per-task={sbatch.cpus_per_task}",
        f"#SBATCH --mem={sbatch.mem}",
        f"#SBATCH --time={sbatch.time}",
        f"#SBATCH --output={log_dir}/slurm-%j.out",
        f"#SBATCH --error={log_dir}/slurm-%j.err",
        "",
    ]

    if config.strict_bash or mode == "smoke":
        lines.extend(["set -euo pipefail", ""])

    if mode == "smoke":
        # A smoke run answers "does the prod configuration start and survive".
        # It can only answer that if it runs in the prod environment, so the
        # single deliberate difference is where W&B logs to. Smoke previously
        # also set PYTHONFAULTHANDLER=1 and two XLA allocator knobs; one of
        # those made habitat_sim SIGABRT in prefill on runs that succeeded
        # under prod (jobs 6056684, 6056813 against 6056750, 2026-07-27), so a
        # green smoke and a red smoke both said nothing about prod.
        lines.extend(['export WANDB_MODE=offline', ""])

    lines.extend([f"mkdir -p {log_dir}", ""])
    lines.extend([
        'echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"',
        "echo \"GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')\"",
        f'GPU_MEMORY_LOG="{log_dir}/gpu-memory-${{SLURM_JOB_ID}}.csv"',
        'if command -v nvidia-smi >/dev/null 2>&1; then',
        '    nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu --format=csv -l 5 > "$GPU_MEMORY_LOG" 2>/dev/null &',
        '    GPU_MONITOR_PID=$!',
        '    trap \'if [ -n "${GPU_MONITOR_PID:-}" ]; then kill "$GPU_MONITOR_PID" 2>/dev/null || true; wait "$GPU_MONITOR_PID" 2>/dev/null || true; fi\' EXIT',
        '    echo "GPU memory log: $GPU_MEMORY_LOG"',
        'fi',
        "",
    ])

    # Define TIMESTAMP only when a rendered value references it.
    referenced = [*args.values(), *config.env.values()]
    if any("${TIMESTAMP}" in str(v) for v in referenced):
        lines.extend(["TIMESTAMP=$(date +%Y%m%d-%H%M%S)", ""])

    if config.env:
        lines.extend(f'export {key}="{value}"' for key, value in config.env.items())
        lines.append("")

    if config.setup:
        lines.extend(config.setup)
        lines.append("")

    if config.curriculum_check:
        lines.extend([
            "# Generate curriculum configs if not present",
            f"if [ ! -f {config.curriculum_check} ]; then",
            '    echo "Generating curriculum configs..."',
            "    uv run python scripts/environments/generate_curriculum.py",
            "fi",
            "",
        ])

    lines.extend(f"# {comment}" for comment in config.comments)

    python_cmd = _python_cmd(config, mode)
    entrypoint = config.script if config.run_id is None else f"{config.script} {config.run_id}"
    arg_lines = [
        line
        for name, value in args.items()
        if (line := _format_arg(name, value, config.arg_style)) is not None
    ]
    if arg_lines:
        lines.append(f"{python_cmd} {entrypoint} \\")
        lines.extend(f"{line} \\" for line in arg_lines[:-1])
        lines.append(arg_lines[-1])
    else:
        lines.append(f"{python_cmd} {entrypoint}")

    if mode == "smoke" and config.smoke.assert_file:
        run_dir = _run_dir(args, config)
        target = f"{run_dir}/{config.smoke.assert_file}"
        lines.extend([
            "",
            f'if [ ! -f "{target}" ]; then',
            f'    echo "[FAIL] {config.smoke.assert_file} missing in {run_dir}" >&2',
            "    exit 1",
            "fi",
        ])
        if config.smoke.assert_min_rows is not None:
            lines.extend([
                f'if [ "$(wc -l < "{target}")" -lt {config.smoke.assert_min_rows} ]; then',
                f'    echo "[FAIL] {config.smoke.assert_file} has fewer than '
                f'{config.smoke.assert_min_rows} rows in {run_dir}" >&2',
                "    exit 1",
                "fi",
            ])
        lines.extend([
            "",
            'echo "=== Smoke PASS ==="',
            f'echo "{config.smoke.assert_file} lines: $(wc -l < "{target}")"',
            f'echo "Output: {run_dir}"',
        ])

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant")
    parser.add_argument("--mode", choices=["prod", "smoke"], default="prod")
    parser.add_argument(
        "--time",
        default=None,
        metavar="SLURM_TIME",
        help="override the selected mode's #SBATCH walltime, e.g. 02:00:00",
    )
    parser.add_argument(
        "--partition",
        default=None,
        metavar="SLURM_PARTITION",
        help="override the selected mode's #SBATCH partition, e.g. gpu_h100",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override an env var (repeatable); wins over the YAML default",
    )
    ns = parser.parse_args(argv)

    env_overrides: dict[str, str] = {}
    for item in ns.env:
        if "=" not in item:
            print(f"launch.py: --env expects KEY=VALUE, got {item!r}", file=sys.stderr)
            return 2
        key, value = item.split("=", 1)
        env_overrides[key] = value

    try:
        config = load_config(ns.variant, env_overrides=env_overrides)
    except Exception as exc:
        print(f"launch.py: {exc}", file=sys.stderr)
        return 2

    try:
        rendered = render_sbatch(
            config, mode=ns.mode, time_override=ns.time, partition_override=ns.partition,
        )
    except Exception as exc:
        print(f"launch.py: {exc}", file=sys.stderr)
        return 2
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
