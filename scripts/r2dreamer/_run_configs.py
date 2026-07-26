"""Central registry of R2Dreamer experiment run configurations.

DRY: the per-run ``run_jax_*.py`` shims used to each repeat the same ~5-line
``sys.path`` bootstrap plus a hardcoded ``train(...)`` call differing only in
env/adapter/curriculum/output_dir/wandb_name/wandb_tags. That metadata now lives
here as one ``RUN_CONFIGS`` table, launched by the single ``run.py`` dispatcher::

    uv run python scripts/r2dreamer/run.py <run-id> [train flags...]

Slurm configs select a run via the ``run_id:`` field (rendered as that leading
positional by ``scripts/slurm/launch.py``); ad-hoc / legacy ``*.sbatch`` files
call ``run.py <run-id>`` directly. ``launch_run`` validates the adapter against
``src.adapters.ADAPTERS`` at launch, so a typo - or a variant not yet migrated
to the routed adapter contract - fails fast instead of at train-time.
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
        adapter="rgb",
        curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1",
        wandb_name="r2d-L1-1house-chair",
        wandb_tags=["curriculum", "level1", "1house", "chair-only", "no-goal"],
    ),
    "habitat-l2-cnn": dict(
        env="habitat",
        adapter="rgb",
        curriculum="L2",
        output_dir="output/runs/r2dreamer-curriculum-l2",
        wandb_name="r2d-L2-buffix",
        wandb_tags=["curriculum", "level2", "1house", "6goals", "buffer-fix", "rerun"],
    ),
    "habitat-l3-cnn": dict(
        env="habitat",
        adapter="rgb",
        curriculum="L3",
        output_dir="output/runs/r2dreamer-curriculum-l3",
        wandb_name="r2d-L3-buffix",
        wandb_tags=["curriculum", "level3", "10houses", "chair-only", "buffer-fix", "rerun"],
    ),
    "habitat-l4-cnn": dict(
        env="habitat",
        adapter="rgb",
        curriculum="L4",
        output_dir="output/runs/r2dreamer-curriculum-l4",
        wandb_name="r2d-L4-buffix",
        wandb_tags=["curriculum", "level4", "10houses", "6goals", "buffer-fix", "rerun"],
    ),
    # ── VGGT adapter variants / ablations ───────────────────────────────────
    # L1–L4 ``rgb_pointmap_pose`` — conv(RGB) + MLP over the pooled 37x37 point map
    # concatenated with the camera pose, one replayed vector per step
    # (3D-50/51/52). The routed encoder fuses the two branches with a Dense;
    # the learned gate of the old HybridEncoder is gone.
    "habitat-l1-hybrid": dict(
        env="habitat",
        adapter="rgb_pointmap_pose",
        curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1-hybrid",
        wandb_name="hybrid-cnn-vggt",
        wandb_tags=[
            "curriculum", "level1", "1house", "chair-only", "vggt",
            "hybrid", "cnn", "wp-cp", "3d-encoder", "jax",
        ],
    ),
    "habitat-l2-hybrid": dict(
        env="habitat",
        adapter="rgb_pointmap_pose",
        curriculum="L2",
        output_dir="output/runs/r2dreamer-curriculum-l2-hybrid",
        wandb_name="l2_hybrid_cnn_points",
        wandb_tags=[
            "curriculum", "level2", "1house", "6goals", "vggt",
            "hybrid", "cnn", "wp-cp", "3d-encoder", "jax",
        ],
    ),
    "habitat-l3-hybrid": dict(
        env="habitat",
        adapter="rgb_pointmap_pose",
        curriculum="L3",
        output_dir="output/runs/r2dreamer-curriculum-l3-hybrid",
        wandb_name="l3_hybrid_cnn_points",
        wandb_tags=[
            "curriculum", "level3", "10houses", "chair-only", "vggt",
            "hybrid", "cnn", "wp-cp", "3d-encoder", "jax",
        ],
    ),
    "habitat-l4-hybrid": dict(
        env="habitat",
        adapter="rgb_pointmap_pose",
        curriculum="L4",
        output_dir="output/runs/r2dreamer-curriculum-l4-hybrid",
        wandb_name="l4_hybrid_cnn_points",
        wandb_tags=[
            "curriculum", "level4", "10houses", "6goals", "vggt",
            "hybrid", "cnn", "wp-cp", "3d-encoder", "jax",
        ],
    ),
    # L1 ``rgb_house_cloud_episodes`` — conv(RGB replay) + one PointNet cloud that
    # survives episode boundaries: every frame's world points are appended and
    # the cloud is voxel-downsampled at each episode start. No camera pose, no
    # per-scene separation - the arm that asks whether cross-episode context
    # helps at all. The cloud is the single live (``buffer=False``) field.
    "habitat-l1-vggt-house-context": dict(
        env="habitat",
        adapter="rgb_house_cloud_episodes",
        curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1-vggt-house-context",
        wandb_name="l1_rgb_replay_vggt_house_context",
        wandb_tags=[
            "curriculum", "level1", "1house", "chair-only", "vggt",
            "house-context", "full-token-transformer", "rgb-replay", "live-cache", "bounded-cache",
            "3d-77", "jax", "3d-encoder",
        ],
    ),
    # L1 ``rgb_house_voxels`` — conv(RGB replay) + MLP(camera pose) + PointNet over
    # a live per-scene voxel-deduplicated house map (PERSIST_SCENE, so every
    # episode of one house extends a single map). The cloud is emitted at a
    # fixed 16384x6 so the branch never recompiles as the map grows; it is the
    # single live (``buffer=False``) field.
    "habitat-l1-vggt-hybrid-house-points-pose": dict(
        env="habitat",
        adapter="rgb_house_voxels",
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
    # L1 ``rgb_house_voxels_gnn`` — the arm above with the house cloud routed to the
    # k-NN-GCN branch instead of PointNet. Identical replay fields, voxel
    # accumulator and extractor policy, which is what keeps the two comparable.
    "habitat-l1-gnn-house-points-pose": dict(
        env="habitat",
        adapter="rgb_house_voxels_gnn",
        curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1-gnn-house-points-pose",
        wandb_name="l1_gnn_house_points_pose",
        wandb_tags=[
            "curriculum", "level1", "1house", "chair-only", "vggt",
            "house-points", "live-buffer", "camera-pose-replay",
            "gnn", "knn-graph", "prototype", "jax", "3d-encoder",
        ],
    ),
    # L1–L4 ``rgb_global_tokens`` — conv(RGB replay) + Transformer over the global half
    # of the final VGGT aggregator tokens (1374x1024). No point map: the context
    # is whatever the streaming attention cache carries, which is what this arm
    # measures. Point/camera heads are off. The tokens describe the *current*
    # frame, so they are replayed per step - at 2.8 MB/row, runs of this variant
    # must cap ``--buffer_capacity``.
    "habitat-l1-vggt-house-global-tokens-nogate": dict(
        env="habitat",
        adapter="rgb_global_tokens",
        curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1-vggt-house-global-tokens-nogate",
        wandb_name="l1_rgb_replay_vggt_house_global_tokens_nogate",
        wandb_tags=[
            "curriculum", "level1", "1house", "chair-only", "vggt",
            "house-context", "global-token-transformer", "no-gate", "rgb-replay",
            "live-cache", "bounded-cache", "3d-90", "jax", "3d-encoder",
        ],
    ),
    "habitat-l2-global-tokens": dict(
        env="habitat",
        adapter="rgb_global_tokens",
        curriculum="L2",
        output_dir="output/runs/r2dreamer-curriculum-l2-global-tokens",
        wandb_name="l2_global_tokens",
        wandb_tags=[
            "curriculum", "level2", "1house", "6goals", "vggt",
            "global-token-transformer", "no-gate", "rgb-replay",
            "live-cache", "bounded-cache", "3d-90", "jax", "3d-encoder",
        ],
    ),
    "habitat-l3-global-tokens": dict(
        env="habitat",
        adapter="rgb_global_tokens",
        curriculum="L3",
        output_dir="output/runs/r2dreamer-curriculum-l3-global-tokens",
        wandb_name="l3_global_tokens",
        wandb_tags=[
            "curriculum", "level3", "10houses", "chair-only", "vggt",
            "global-token-transformer", "no-gate", "rgb-replay",
            "live-cache", "bounded-cache", "3d-90", "jax", "3d-encoder",
        ],
    ),
    "habitat-l4-global-tokens": dict(
        env="habitat",
        adapter="rgb_global_tokens",
        curriculum="L4",
        output_dir="output/runs/r2dreamer-curriculum-l4-global-tokens",
        wandb_name="l4_global_tokens",
        wandb_tags=[
            "curriculum", "level4", "10houses", "6goals", "vggt",
            "global-token-transformer", "no-gate", "rgb-replay",
            "live-cache", "bounded-cache", "3d-90", "jax", "3d-encoder",
        ],
    ),
    # L1 ``rgb_full_tokens`` — the arm above with both halves of the aggregator
    # tokens (1374x2048) instead of the global half only, so it measures what the
    # frame half adds. At 5.6 MB/row it needs a tighter ``--buffer_capacity``.
    "habitat-l1-full-tokens": dict(
        env="habitat",
        adapter="rgb_full_tokens",
        curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1-rgb-full-tokens",
        wandb_name="l1_rgb_full_tokens",
        wandb_tags=[
            "curriculum", "level1", "1house", "chair-only", "vggt",
            "full-token-transformer", "no-gate", "rgb-replay",
            "live-cache", "bounded-cache", "3d-77", "jax", "3d-encoder",
        ],
    ),
    # L1–L4 ``aggregator_pooled`` — the cheap end of the token family: the global
    # token half pooled to [camera, patch mean, patch max] (3072) through an MLP
    # branch. No appearance channel, no geometry, 12 KB/row.
    "habitat-l1-aggregator-pooled": dict(
        env="habitat",
        adapter="aggregator_pooled",
        curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1-aggregator-pooled",
        wandb_name="l1_aggregator_pooled",
        wandb_tags=[
            "curriculum", "level1", "1house", "chair-only", "vggt",
            "aggregator-pooled", "pool-on-device", "skip-heads",
            "jax", "3d-encoder",
        ],
    ),
    "habitat-l2-aggregator-pooled": dict(
        env="habitat",
        adapter="aggregator_pooled",
        curriculum="L2",
        output_dir="output/runs/r2dreamer-curriculum-l2-aggregator-pooled",
        wandb_name="l2_aggregator_pooled",
        wandb_tags=[
            "curriculum", "level2", "1house", "6goals", "vggt",
            "aggregator-pooled", "pool-on-device", "skip-heads",
            "jax", "3d-encoder",
        ],
    ),
    "habitat-l3-aggregator-pooled": dict(
        env="habitat",
        adapter="aggregator_pooled",
        curriculum="L3",
        output_dir="output/runs/r2dreamer-curriculum-l3-aggregator-pooled",
        wandb_name="l3_aggregator_pooled",
        wandb_tags=[
            "curriculum", "level3", "10houses", "chair-only", "vggt",
            "aggregator-pooled", "pool-on-device", "skip-heads",
            "jax", "3d-encoder",
        ],
    ),
    "habitat-l4-aggregator-pooled": dict(
        env="habitat",
        adapter="aggregator_pooled",
        curriculum="L4",
        output_dir="output/runs/r2dreamer-curriculum-l4-aggregator-pooled",
        wandb_name="l4_aggregator_pooled",
        wandb_tags=[
            "curriculum", "level4", "10houses", "6goals", "vggt",
            "aggregator-pooled", "pool-on-device", "skip-heads",
            "jax", "3d-encoder",
        ],
    ),
    # ── Geometry-only arms (no appearance channel, hence no decoder target) ──
    # L1–L4 ``pointmap_pose`` — the ``rgb_pointmap_pose`` arm above with the image
    # field removed: only the 37x37 pooled point map plus camera pose, through an
    # MLP branch. The pair isolates what appearance contributes.
    "habitat-l1-pointmap-pose": dict(
        env="habitat",
        adapter="pointmap_pose",
        curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1-pointmap-pose",
        wandb_name="l1_pointmap_pose",
        wandb_tags=[
            "curriculum", "level1", "1house", "chair-only", "vggt",
            "pointmap-pose", "geometry-only", "3d-52", "jax", "3d-encoder",
        ],
    ),
    "habitat-l2-pointmap-pose": dict(
        env="habitat",
        adapter="pointmap_pose",
        curriculum="L2",
        output_dir="output/runs/r2dreamer-curriculum-l2-pointmap-pose",
        wandb_name="l2_pointmap_pose",
        wandb_tags=[
            "curriculum", "level2", "1house", "6goals", "vggt",
            "pointmap-pose", "geometry-only", "3d-52", "jax", "3d-encoder",
        ],
    ),
    "habitat-l3-pointmap-pose": dict(
        env="habitat",
        adapter="pointmap_pose",
        curriculum="L3",
        output_dir="output/runs/r2dreamer-curriculum-l3-pointmap-pose",
        wandb_name="l3_pointmap_pose",
        wandb_tags=[
            "curriculum", "level3", "10houses", "chair-only", "vggt",
            "pointmap-pose", "geometry-only", "3d-52", "jax", "3d-encoder",
        ],
    ),
    "habitat-l4-pointmap-pose": dict(
        env="habitat",
        adapter="pointmap_pose",
        curriculum="L4",
        output_dir="output/runs/r2dreamer-curriculum-l4-pointmap-pose",
        wandb_name="l4_pointmap_pose",
        wandb_tags=[
            "curriculum", "level4", "10houses", "6goals", "vggt",
            "pointmap-pose", "geometry-only", "3d-52", "jax", "3d-encoder",
        ],
    ),
    # L1 ``pointmap_pose_64`` — resolution ablation of the arm above: identical
    # pipeline, point map reduced to 64x64 instead of the 37x37 patch grid.
    "habitat-l1-pointmap-pose-64": dict(
        env="habitat",
        adapter="pointmap_pose_64",
        curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1-pointmap-pose-64",
        wandb_name="l1_pointmap_pose_64",
        wandb_tags=[
            "curriculum", "level1", "1house", "chair-only", "vggt",
            "pointmap-pose", "pointmap-64", "geometry-only", "3d-52",
            "jax", "3d-encoder",
        ],
    ),
    # L1 ``pointmap_dense`` — the unpooled 518x518x3 point map through a conv
    # branch, so the geometry keeps its spatial structure. At 1.6 MB/row runs of
    # this variant must cap ``--buffer_capacity``.
    "habitat-l1-pointmap-dense": dict(
        env="habitat",
        adapter="pointmap_dense",
        curriculum="L1",
        output_dir="output/runs/r2dreamer-curriculum-l1-pointmap-dense",
        wandb_name="l1_pointmap_dense",
        wandb_tags=[
            "curriculum", "level1", "1house", "chair-only", "vggt",
            "pointmap-dense", "full-res-518", "conv-points", "geometry-only",
            "3d-53", "jax", "3d-encoder",
        ],
    ),
    # ── Crafter (non-curriculum sanity env) ─────────────────────────────────
    "crafter-cnn": dict(
        env="crafter",
        adapter="rgb",
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

    # Fail fast on an adapter typo (or a variant not yet migrated to the routed
    # adapter contract), against the canonical registry.
    from src.adapters import ADAPTERS

    if cfg["adapter"] not in ADAPTERS:
        raise KeyError(
            f"Run {name!r} uses unknown adapter {cfg['adapter']!r}. "
            f"Available: {sorted(ADAPTERS)}"
        )

    from src.main import train

    return train(**cfg, argv=argv)
