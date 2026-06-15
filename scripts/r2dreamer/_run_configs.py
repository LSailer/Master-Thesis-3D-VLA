"""Central registry of R2Dreamer experiment run configurations.

DRY: the per-run ``run_jax_*.py`` shims used to each repeat the same ~5-line
``sys.path`` bootstrap plus a hardcoded ``train(...)`` call differing only in
env/Observation Preparation/curriculum/output_dir/wandb_name/wandb_tags. That metadata now lives
here as one ``RUN_CONFIGS`` table, launched by the single ``run.py`` dispatcher::

    uv run python scripts/r2dreamer/run.py <run-id> [train flags...]

Slurm configs select a run via the ``run_id:`` field (rendered as that leading
positional by ``scripts/slurm/launch.py``); ad-hoc / legacy ``*.sbatch`` files
call ``run.py <run-id>`` directly. ``launch_run`` validates the Observation
Preparation mode against the canonical registry at launch, so a typo fails fast
instead of at train-time.
"""

from __future__ import annotations

import os
import sys
from typing import Any

# Make ``src`` importable regardless of the CWD the shim is launched from.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# Each value is the full kwargs forwarded to ``src.main.train`` for one run.
RUN_CONFIGS: dict[str, dict[str, Any]] = {
    # ── CNN curriculum baselines (L1–L4) ────────────────────────────────────
    "habitat-l1-cnn": dict(
        env="habitat",
        observation_preparation="cnn",
        curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1",
        wandb_name="r2d-L1-1house-chair",
        wandb_tags=["curriculum", "level1", "1house", "chair-only", "no-goal"],
    ),
    "habitat-l2-cnn": dict(
        env="habitat",
        observation_preparation="cnn",
        curriculum="L2",
        output_dir="output/runs/r2dreamer-curriculum-l2",
        wandb_name="r2d-L2-buffix",
        wandb_tags=["curriculum", "level2", "1house", "6goals", "buffer-fix", "rerun"],
    ),
    "habitat-l3-cnn": dict(
        env="habitat",
        observation_preparation="cnn",
        curriculum="L3",
        output_dir="output/runs/r2dreamer-curriculum-l3",
        wandb_name="r2d-L3-buffix",
        wandb_tags=["curriculum", "level3", "10houses", "chair-only", "buffer-fix", "rerun"],
    ),
    "habitat-l4-cnn": dict(
        env="habitat",
        observation_preparation="cnn",
        curriculum="L4",
        output_dir="output/runs/r2dreamer-curriculum-l4",
        wandb_name="r2d-L4-buffix",
        wandb_tags=["curriculum", "level4", "10houses", "6goals", "buffer-fix", "rerun"],
    ),
    # ── VGGT Observation Preparation curriculum (L1–L4) ─────────────────────
    "habitat-l1-vggt": dict(
        env="habitat",
        observation_preparation="vggt",
        curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1-vggt",
        wandb_name="vggt_jax",
        wandb_tags=[
            "curriculum", "level1", "1house", "chair-only", "vggt",
            "vggt_jax", "jax", "3d-encoder",
        ],
    ),
    "habitat-l2-vggt": dict(
        env="habitat",
        observation_preparation="vggt",
        curriculum="L2",
        output_dir="output/runs/r2dreamer-curriculum-l2-vggt",
        wandb_name="r2d-L2-vggt",
        wandb_tags=["curriculum", "level2", "1house", "6goals", "vggt", "jax", "3d-encoder"],
    ),
    "habitat-l3-vggt": dict(
        env="habitat",
        observation_preparation="vggt",
        curriculum="L3",
        output_dir="output/runs/r2dreamer-curriculum-l3-vggt",
        wandb_name="r2d-L3-vggt",
        wandb_tags=["curriculum", "level3", "10houses", "chair-only", "vggt", "jax", "3d-encoder"],
    ),
    "habitat-l4-vggt": dict(
        env="habitat",
        observation_preparation="vggt",
        curriculum="L4",
        output_dir="output/runs/r2dreamer-curriculum-l4-vggt",
        wandb_name="r2d-L4-vggt",
        wandb_tags=["curriculum", "level4", "10houses", "6goals", "vggt", "jax", "3d-encoder"],
    ),
    # ── VGGT Observation Preparation variants / ablations ───────────────────
    # L1 Hybrid — CNN(RGB) + gated MLP(WP/CP) hybrid Encoder Module (3D-50/51/52).
    "habitat-l1-hybrid": dict(
        env="habitat",
        observation_preparation="hybrid",
        curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1-hybrid",
        wandb_name="hybrid-cnn-vggt",
        wandb_tags=[
            "curriculum", "level1", "1house", "chair-only", "vggt",
            "hybrid", "cnn", "wp-cp", "3d-encoder", "jax",
        ],
    ),
    # L1 VGGT — WP+CP MLP at a 64x64 world-point grid (3D-52/3D-53).
    "habitat-l1-vggt-wp-cp-64": dict(
        env="habitat",
        observation_preparation="vggt_wp_cp_64",
        curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1-vggt-wp-cp-64",
        wandb_name="wp-cp-mlp-64",
        wandb_tags=[
            "curriculum", "level1", "1house", "chair-only", "vggt",
            "wp-cp", "mlp-3layer", "wp-64", "3d-52", "jax", "3d-encoder",
        ],
    ),
    # L1 VGGT — full-resolution 518x518x3 world-point CNN encoder (3D-53).
    "habitat-l1-vggt-wp-dense": dict(
        env="habitat",
        observation_preparation="vggt_wp_dense_cnn",
        curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1-vggt-wp-dense",
        wandb_name="vggt_wp_dense_cnn",
        wandb_tags=[
            "curriculum", "level1", "1house", "chair-only", "vggt",
            "wp-dense", "full-res-518", "cnn", "jax", "3d-encoder",
        ],
    ),
    # L1 VGGT aggregator-MLP encoder (variant-1).
    "habitat-l1-vggt-aggregator-mlp": dict(
        env="habitat",
        observation_preparation="vggt_aggregator_mlp",
        curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1-vggt-aggregator-mlp",
        wandb_name="variant-1-aggregator-mlp",
        wandb_tags=[
            "curriculum", "level1", "1house", "chair-only", "vggt",
            "aggregator-mlp", "variant-1", "jax", "3d-encoder",
        ],
    ),
    # ── Crafter (non-curriculum sanity env) ─────────────────────────────────
    "crafter-cnn": dict(
        env="crafter",
        observation_preparation="cnn",
        curriculum=None,
        output_dir="output/runs/r2dreamer-crafter",
        wandb_name="r2d-crafter",
        wandb_tags=["crafter", "cnn"],
    ),
}


def launch_run(name: str, *, argv: list[str] | None = None):
    """Validate and launch the named run configuration via ``src.main.train``.

    ``argv`` is forwarded to ``train`` (and thence to argparse); the generic
    ``run.py`` dispatcher passes the train flags that followed the run id so
    they do not have to be re-parsed out of ``sys.argv`` (which still holds the
    run id positional). When ``None``, ``train`` falls back to ``sys.argv[1:]``.
    """
    if name not in RUN_CONFIGS:
        raise KeyError(f"Unknown run {name!r}. Available: {sorted(RUN_CONFIGS)}")
    cfg = RUN_CONFIGS[name]

    # Fail fast on an Observation Preparation typo, against the canonical registry.
    from src.r2dreamer.launch.registries import observation_preparation_registry

    if cfg["observation_preparation"] not in observation_preparation_registry:
        raise KeyError(
            f"Run {name!r} uses unknown observation_preparation "
            f"{cfg['observation_preparation']!r}. "
            f"Available: {sorted(observation_preparation_registry)}"
        )

    from src.main import train

    return train(**cfg, argv=argv)
