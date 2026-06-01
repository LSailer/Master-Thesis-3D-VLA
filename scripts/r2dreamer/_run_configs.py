"""Central registry of R2Dreamer experiment run configurations.

DRY: the ``run_jax_*.py`` shims used to each repeat the same ~5-line ``sys.path``
bootstrap plus a hardcoded ``train(...)`` call differing only in
env/encoder/curriculum/output_dir/wandb_name/wandb_tags. That metadata now lives
here as one ``RUN_CONFIGS`` table, and each shim collapses to::

    import _run_configs
    if __name__ == "__main__":
        _run_configs.launch_run("habitat-l1-hybrid")

One file per run id is kept (rather than a single CLI) so the ``script:`` paths
referenced by ``scripts/slurm/configs/*.yaml`` stay valid. ``launch_run`` also
validates the encoder against the canonical ``encoder_registry`` at launch, so a
typo fails fast instead of at train-time.
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
    # L1 Hybrid — CNN(RGB) + gated MLP(WP/CP) hybrid encoder (3D-50/51/52).
    "habitat-l1-hybrid": dict(
        env="habitat",
        encoder="hybrid",
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
        encoder="vggt_wp_cp_64",
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
        encoder="vggt_wp_dense_cnn",
        curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1-vggt-wp-dense",
        wandb_name="vggt_wp_dense_cnn",
        wandb_tags=[
            "curriculum", "level1", "1house", "chair-only", "vggt",
            "wp-dense", "full-res-518", "cnn", "jax", "3d-encoder",
        ],
    ),
}


def launch_run(name: str):
    """Validate and launch the named run configuration via ``src.main.train``."""
    if name not in RUN_CONFIGS:
        raise KeyError(f"Unknown run {name!r}. Available: {sorted(RUN_CONFIGS)}")
    cfg = RUN_CONFIGS[name]

    # Fail fast on an encoder typo, against the canonical registry.
    from src.r2dreamer.launch.registries import encoder_registry

    if cfg["encoder"] not in encoder_registry:
        raise KeyError(
            f"Run {name!r} uses unknown encoder {cfg['encoder']!r}. "
            f"Available: {sorted(encoder_registry)}"
        )

    from src.main import train

    return train(**cfg)
