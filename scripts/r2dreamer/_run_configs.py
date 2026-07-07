"""Central registry of R2Dreamer experiment run configurations.

DRY: the per-run ``run_jax_*.py`` shims used to each repeat the same ~5-line
``sys.path`` bootstrap plus a hardcoded ``train(...)`` call differing only in
env/encoder/curriculum/output_dir/wandb_name/wandb_tags. That metadata now lives
here as one ``RUN_CONFIGS`` table, launched by the single ``run.py`` dispatcher::

    uv run python scripts/r2dreamer/run.py <run-id> [train flags...]

Slurm configs select a run via the ``run_id:`` field (rendered as that leading
positional by ``scripts/slurm/launch.py``); ad-hoc / legacy ``*.sbatch`` files
call ``run.py <run-id>`` directly. ``launch_run`` validates the encoder against
the canonical ``encoder_registry`` at launch, so a typo fails fast instead of at
train-time.
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
        encoder="cnn",
        curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1",
        wandb_name="r2d-L1-1house-chair",
        wandb_tags=["curriculum", "level1", "1house", "chair-only", "no-goal"],
    ),
    "habitat-l2-cnn": dict(
        env="habitat",
        encoder="cnn",
        curriculum="L2",
        output_dir="output/runs/r2dreamer-curriculum-l2",
        wandb_name="r2d-L2-buffix",
        wandb_tags=["curriculum", "level2", "1house", "6goals", "buffer-fix", "rerun"],
    ),
    "habitat-l3-cnn": dict(
        env="habitat",
        encoder="cnn",
        curriculum="L3",
        output_dir="output/runs/r2dreamer-curriculum-l3",
        wandb_name="r2d-L3-buffix",
        wandb_tags=["curriculum", "level3", "10houses", "chair-only", "buffer-fix", "rerun"],
    ),
    "habitat-l4-cnn": dict(
        env="habitat",
        encoder="cnn",
        curriculum="L4",
        output_dir="output/runs/r2dreamer-curriculum-l4",
        wandb_name="r2d-L4-buffix",
        wandb_tags=["curriculum", "level4", "10houses", "6goals", "buffer-fix", "rerun"],
    ),
    # ── VGGT 3D-encoder curriculum (L1–L4) ──────────────────────────────────
    "habitat-l1-vggt": dict(
        env="habitat",
        encoder="vggt",
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
        encoder="vggt",
        curriculum="L2",
        output_dir="output/runs/r2dreamer-curriculum-l2-vggt",
        wandb_name="r2d-L2-vggt",
        wandb_tags=["curriculum", "level2", "1house", "6goals", "vggt", "jax", "3d-encoder"],
    ),
    "habitat-l3-vggt": dict(
        env="habitat",
        encoder="vggt",
        curriculum="L3",
        output_dir="output/runs/r2dreamer-curriculum-l3-vggt",
        wandb_name="r2d-L3-vggt",
        wandb_tags=["curriculum", "level3", "10houses", "chair-only", "vggt", "jax", "3d-encoder"],
    ),
    "habitat-l4-vggt": dict(
        env="habitat",
        encoder="vggt",
        curriculum="L4",
        output_dir="output/runs/r2dreamer-curriculum-l4-vggt",
        wandb_name="r2d-L4-vggt",
        wandb_tags=["curriculum", "level4", "10houses", "6goals", "vggt", "jax", "3d-encoder"],
    ),
    # ── VGGT encoder variants / ablations ───────────────────────────────────
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
    # L1 House Context — RGB replay + live full-token InfiniteVGGT scene memory.
    "habitat-l1-vggt-house-context": dict(
        env="habitat",
        encoder="vggt_house_context",
        curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1-vggt-house-context",
        wandb_name="l1_rgb_replay_vggt_house_context",
        wandb_tags=[
            "curriculum", "level1", "1house", "chair-only", "vggt",
            "house-context", "full-token-transformer", "rgb-replay", "live-cache", "bounded-cache",
            "3d-77", "jax", "3d-encoder",
        ],
    ),
    # L1 house points + current VGGT camera pose; PointNet house branch
    # (src/r2dreamer/encoders/pointnet.py).
    "habitat-l1-vggt-house-points-pose": dict(
        env="habitat",
        encoder="vggt_house_points_pose",
        curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1-vggt-house-points-pose",
        wandb_name="l1_static_house_points_pose",
        wandb_tags=[
            "curriculum", "level1", "1house", "chair-only", "vggt",
            "house-points", "static-sidecar", "camera-pose-replay",
            "pointnet", "tnet", "jax", "3d-encoder",
        ],
    ),
    # L1 additive hybrid: rgb64 CNN backbone + zero-init-gated live house
    # points/pose branches (starts exactly at the CNN baseline).
    "habitat-l1-vggt-hybrid-house-points-pose": dict(
        env="habitat",
        encoder="vggt_hybrid_house_points_pose",
        curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1-vggt-hybrid-house-points-pose",
        wandb_name="l1_hybrid_house_points_pose",
        wandb_tags=[
            "curriculum", "level1", "1house", "chair-only", "vggt",
            "house-points", "live-buffer", "camera-pose-replay",
            "point-mlp", "cnn-backbone", "additive-hybrid", "gated",
            "jax", "3d-encoder",
        ],
    ),
    # L1 live house points + GNN house branch (src/r2dreamer/encoders/gnn_house.py).
    "habitat-l1-gnn-house-points-pose": dict(
        env="habitat",
        encoder="gnn_house_points_pose",
        curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1-gnn-house-points-pose",
        wandb_name="l1_gnn_house_points_pose",
        wandb_tags=[
            "curriculum", "level1", "1house", "chair-only", "vggt",
            "house-points", "live-buffer", "camera-pose-replay",
            "gnn", "knn-graph", "prototype", "jax", "3d-encoder",
        ],
    ),
    # L1 live house points + EdgeConv-variant GNN house branch (gnn_house.py).
    "habitat-l1-gnn-edge-house-points-pose": dict(
        env="habitat",
        encoder="gnn_edge_house_points_pose",
        curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1-gnn-edge-house-points-pose",
        wandb_name="l1_gnn_edge_house_points_pose",
        wandb_tags=[
            "curriculum", "level1", "1house", "chair-only", "vggt",
            "house-points", "live-buffer", "camera-pose-replay",
            "gnn", "knn-graph", "edgeconv", "residual", "prototype",
            "jax", "3d-encoder",
        ],
    ),
    # L1 live house points + classic PointNet house branch (encoders/pointnet.py).
    "habitat-l1-pointnet-house-points-pose": dict(
        env="habitat",
        encoder="pointnet",
        curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1-pointnet-house-points-pose",
        wandb_name="l1_pointnet_house_points_pose",
        wandb_tags=[
            "curriculum", "level1", "1house", "chair-only", "vggt",
            "house-points", "live-buffer", "camera-pose-replay",
            "pointnet", "tnet", "jax", "3d-encoder",
        ],
    ),
    # L1 full-token no-gate — RGB replay + live full-token Transformer inside agent.
    "habitat-l1-vggt-house-full-tokens-nogate": dict(
        env="habitat",
        encoder="vggt_house_full_tokens_nogate",
        curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1-vggt-house-full-tokens-nogate",
        wandb_name="l1_rgb_replay_vggt_house_full_tokens_nogate",
        wandb_tags=[
            "curriculum", "level1", "1house", "chair-only", "vggt",
            "house-context", "full-token-transformer", "no-gate", "rgb-replay",
            "live-cache", "bounded-cache", "3d-77", "jax", "3d-encoder",
        ],
    ),
    # L1 global-token no-gate — RGB replay + singleton live global-token context.
    "habitat-l1-vggt-house-global-tokens-nogate": dict(
        env="habitat",
        encoder="vggt_house_global_tokens_nogate",
        curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1-vggt-house-global-tokens-nogate",
        wandb_name="l1_rgb_replay_vggt_house_global_tokens_nogate",
        wandb_tags=[
            "curriculum", "level1", "1house", "chair-only", "vggt",
            "house-context", "global-token-transformer", "no-gate", "rgb-replay",
            "live-cache", "bounded-cache", "3d-90", "jax", "3d-encoder",
        ],
    ),
    # L1 house global embedding — RGB replay + split VGGT global tokens fed to
    # a PointNet reducer (max-pool over 1369 patch tokens; camera token on its
    # own side branch). PERSIST_SCENE, heads off (src/prototyp/
    # house_global_embedding/IDEA.md).
    "habitat-l1-vggt-house-global-embedding": dict(
        env="habitat",
        encoder="vggt_house_global_embedding",
        curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1-vggt-house-global-embedding",
        wandb_name="l1_rgb_replay_vggt_house_global_embedding",
        wandb_tags=[
            "curriculum", "level1", "1house", "chair-only", "vggt",
            "house-context", "global-embedding", "pointnet-reducer", "rgb-replay",
            "live-cache", "persist-scene", "heads-off", "jax", "3d-encoder",
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
    # L1 VGGT — 64x64 world-point CNN + camera-pose MLP encoder (3D-89).
    "habitat-l1-vggt-wp64-cnn-cp-mlp": dict(
        env="habitat",
        encoder="vggt_wp64_cnn_cp_mlp",
        curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1-vggt-wp64-cnn-cp-mlp",
        wandb_name="wp64-cnn-cp-mlp",
        wandb_tags=[
            "curriculum", "level1", "1house", "chair-only", "vggt",
            "wp-64", "wp-cnn", "cp-mlp", "3d-89", "jax", "3d-encoder",
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
    # L1 VGGT aggregator-MLP encoder (variant-1).
    "habitat-l1-vggt-aggregator-mlp": dict(
        env="habitat",
        encoder="vggt_aggregator_mlp",
        curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1-vggt-aggregator-mlp",
        wandb_name="variant-1-aggregator-mlp",
        wandb_tags=[
            "curriculum", "level1", "1house", "chair-only", "vggt",
            "aggregator-mlp", "variant-1", "jax", "3d-encoder",
        ],
    ),
    # L1 VGGT full aggregator-token Transformer encoder (3D-75).
    "habitat-l1-vggt-agg-token-transformer": dict(
        env="habitat",
        encoder="vggt_agg_token_transformer",
        curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1-vggt-agg-token-transformer",
        wandb_name="vggt-agg-token-transformer",
        wandb_tags=[
            "curriculum", "level1", "1house", "chair-only", "vggt",
            "aggregator-tokens", "token-transformer", "3d-75", "jax", "3d-encoder",
        ],
    ),
    # ── Crafter (non-curriculum sanity env) ─────────────────────────────────
    "crafter-cnn": dict(
        env="crafter",
        encoder="cnn",
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

    # Fail fast on an encoder typo, against the canonical registry.
    from src.r2dreamer.launch.registries import encoder_registry

    if cfg["encoder"] not in encoder_registry:
        raise KeyError(
            f"Run {name!r} uses unknown encoder {cfg['encoder']!r}. "
            f"Available: {sorted(encoder_registry)}"
        )

    from src.main import train

    return train(**cfg, argv=argv)
