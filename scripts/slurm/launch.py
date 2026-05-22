#!/usr/bin/env python3
"""Render and validate YAML-backed Slurm launch configs."""

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
ARG_ORDER = (
    "steps",
    "prefill",
    "checkpoint_every",
    "output_dir",
    "seed",
    "log_every",
    "wandb_project",
    "wandb_name",
    "wandb_tags",
    "render_resolution",
)
QUOTE_ARGS = {"output_dir", "seed", "wandb_name", "wandb_tags"}


class SbatchConfig(BaseModel):
    """SBATCH directives shared by all modes."""

    model_config = ConfigDict(extra="forbid")

    partition: str
    gres: str
    ntasks: int
    cpus_per_task: int
    mem: str
    time: str


class ArgsConfig(BaseModel):
    """Training arguments forwarded to the Python training entrypoint."""

    model_config = ConfigDict(extra="allow")

    steps: int = Field(..., gt=0)
    prefill: int = Field(..., ge=0)
    checkpoint_every: int = Field(..., gt=0)
    output_dir: str
    seed: str | int
    log_every: int = Field(..., gt=0)
    wandb_project: str
    wandb_name: str
    wandb_tags: str
    render_resolution: int = Field(..., gt=0)


class SmokeConfig(BaseModel):
    """Mode-specific overrides for dev-cluster smoke submissions."""

    model_config = ConfigDict(extra="forbid")

    partition: str = "dev_gpu_h100"
    time: str = "00:30:00"
    args: dict[str, Any] = Field(default_factory=dict)


class LaunchConfig(BaseModel):
    """Fully merged launch config."""

    model_config = ConfigDict(extra="forbid")

    job_name: str
    output_dir: str
    script: str
    sbatch: SbatchConfig
    args: ArgsConfig
    comments: list[str] = Field(default_factory=list)
    smoke: SmokeConfig = Field(default_factory=SmokeConfig)

    @field_validator("script")
    @classmethod
    def script_must_be_relative(cls, value: str) -> str:
        if Path(value).is_absolute():
            raise ValueError("script must be repo-relative")
        return value


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
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


def load_config(name: str) -> LaunchConfig:
    """Load, merge, and validate a launch config by variant name."""

    if "/" in name or name.startswith("."):
        raise ValueError(f"Invalid variant name: {name!r}")

    raw = _read_yaml(CONFIG_DIR / f"{name}.yaml")
    parent = raw.pop("extends", None)
    if parent:
        parent_name = str(parent)
        if not parent_name.endswith(".yaml"):
            parent_name = f"{parent_name}.yaml"
        raw = _deep_merge(_read_yaml(CONFIG_DIR / parent_name), raw)

    try:
        return LaunchConfig.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid Slurm config for {name!r}:\n{exc}") from exc


def _mode_config(config: LaunchConfig, mode: Literal["prod", "smoke"]) -> tuple[SbatchConfig, dict[str, Any]]:
    sbatch = config.sbatch.model_copy(deep=True)
    args = dict(config.args.model_dump())
    if mode == "smoke":
        sbatch.partition = config.smoke.partition
        sbatch.time = config.smoke.time
        args.update(config.smoke.args)
    return sbatch, args


def _format_arg(name: str, value: Any) -> str:
    rendered = str(value)
    if name in QUOTE_ARGS:
        rendered = f'"{rendered}"'
    return f"    --{name} {rendered}"


def render_sbatch(config: LaunchConfig, *, mode: Literal["prod", "smoke"] = "prod") -> str:
    """Render an sbatch script."""

    sbatch, args = _mode_config(config, mode)
    output_dir = config.output_dir if mode == "prod" else f"{config.output_dir}/smoke"
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
        f"#SBATCH --output={output_dir}/slurm-%j.out",
        f"#SBATCH --error={output_dir}/slurm-%j.err",
        "",
    ]

    if mode == "smoke":
        lines.extend([
            "set -euo pipefail",
            "",
            "export WANDB_MODE=offline",
            "export PYTHONFAULTHANDLER=1",
            # Reduce JAX/habitat CUDA contention on short queues
            "export XLA_PYTHON_CLIENT_PREALLOCATE=false",
            "export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7",
            "",
        ])

    lines.extend([
        f"mkdir -p {output_dir}",
        "",
        'echo "Job $SLURM_JOB_ID on $(hostname) at $(date)"',
        'echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo \'N/A\')"',
        "",
        "# Generate curriculum configs if not present",
        "if [ ! -f data/curriculum/level1_1house_1goal.json ]; then",
        '    echo "Generating curriculum configs..."',
        "    uv run python scripts/environments/generate_curriculum.py",
        "fi",
        "",
    ])

    for comment in config.comments:
        lines.append(f"# {comment}")

    train_cmd = "uv run --no-sync python" if mode == "smoke" else "uv run python"
    lines.append(f"{train_cmd} {config.script} \\")
    arg_lines = [_format_arg(name, args[name]) for name in ARG_ORDER]
    for line in arg_lines[:-1]:
        lines.append(f"{line} \\")
    lines.append(arg_lines[-1])

    if mode == "smoke":
        smoke_output = args["output_dir"]
        lines.extend([
            "",
            f'if [ ! -f "{smoke_output}/metrics.csv" ]; then',
            f'    echo "[FAIL] metrics.csv missing in {smoke_output}" >&2',
            "    exit 1",
            "fi",
            f'if [ "$(wc -l < "{smoke_output}/metrics.csv")" -lt 5 ]; then',
            f'    echo "[FAIL] metrics.csv has fewer than 5 rows in {smoke_output}" >&2',
            "    exit 1",
            "fi",
            "",
            'echo "=== Smoke PASS ==="',
            f'echo "metrics.csv lines: $(wc -l < "{smoke_output}/metrics.csv")"',
            f'echo "Output: {smoke_output}"',
        ])

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant")
    parser.add_argument("--mode", choices=["prod", "smoke"], default="prod")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.variant)
    except Exception as exc:
        print(f"launch.py: {exc}", file=sys.stderr)
        return 2

    print(render_sbatch(config, mode=args.mode), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
