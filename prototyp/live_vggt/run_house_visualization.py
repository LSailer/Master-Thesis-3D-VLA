"""Collect VGGT point-head predictions from random walks into one house cloud.

Successor of the archived three-arm ablation driver
(``src/prototyp/__archiv__/global_token_reconstruction/``): the same
random-agent + PERSIST_SCENE extractor stream, optionally over multiple
episodes of the same scene (``--episodes``). Each frame's confident,
finite points are moved to compact host arrays (float32 xyz + uint8 rgb)
as they are produced, the accumulated cloud is voxel-downsampled in JAX
(``--down-cells``, extent-relative) and exported as ``final.ply`` via the
direct NumPy PLY writer. A MANIFEST.json records ok/failed status —
habitat GL teardown poisons SLURM exit codes on this cluster, so the
manifest is the ground truth for run success.

Run on a GPU node:
    .venv/bin/python prototyp/live_vggt/run_house_visualization.py \
        --steps 500 --seed 42
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp


# Video frames and accumulated points are host-side copies; NumPy marks
# the device -> host boundary.
import numpy as np
from PIL import Image as PILImage

from src.baselines.random_agent import RandomAgent
from src.r2dreamer.encoders.base import VGGTEncoder
from src.r2dreamer.launch.habitat_setup import make_habitat_env
from src.r2dreamer.manifest import write_manifest_end, write_manifest_start
from src.shared.ply_io import (
    flattened_xyz_rgb,
    write_world_points_ply,
)
from src.shared.pointcloud import voxel_down_sample
from src.shared.video_utils import write_frames_mp4
from src.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor, ResetMode

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the house-visualization collection run.

    Returns:
        The parsed argparse namespace.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--steps", type=int, default=500, help="Random-walk frames per episode."
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=1,
        help=(
            "Episodes to walk in the SAME scene; each starts with env.reset() "
            "and keeps accumulating into one cloud (PERSIST_SCENE keeps the "
            "VGGT stream, so all episodes share one world frame)."
        ),
    )
    p.add_argument("--seed", type=int, default=42, help="Env + action-order seed.")
    p.add_argument("--curriculum", type=str, default="L1")
    p.add_argument(
        "--confidence-score",
        type=float,
        default=1.5,
        help="Minimum VGGT confidence for admitting points (live-path default).",
    )
    p.add_argument(
        "--down-voxel",
        type=float,
        default=None,
        help=(
            "Downsample voxel edge in VGGT normalized units (NOT meters: "
            "VGGT output is scene-scale-normalized). Overrides --down-cells."
        ),
    )
    p.add_argument(
        "--down-cells",
        type=int,
        default=1000,
        help=(
            "Scene-relative resolution: voxel edge = max cloud extent / N. "
            "Ignored when --down-voxel is set. Default 1000 keeps a dense "
            "visualization cloud; note the bf16 point-head lattice "
            "(~0.008 units) bounds the real resolvable detail."
        ),
    )
    p.add_argument(
        "--output",
        type=str,
        default="outputs/prototype/live_vggt",
        help="Output root; a per-run subdirectory is created inside.",
    )
    p.add_argument(
        "--video-fps",
        type=int,
        default=10,
        help="Write observations.mp4 of the walked frames (0 disables).",
    )
    return p.parse_args()


def collect_and_export(args: argparse.Namespace, out_dir: Path) -> None:
    """Stream the random walks, accumulate host-side points, export final.ply.

    Args:
      args: Parsed CLI namespace from :func:`parse_args`.
      out_dir: Per-run output directory (already created).

    Returns:
      None.
    """
    t_env = time.perf_counter()
    env = make_habitat_env(
        curriculum=args.curriculum,
        render_resolution=518,
        seed=args.seed,
    )
    frame = env.reset()
    print(
        f"[house-viz] env ready in {time.perf_counter() - t_env:.1f}s "
        f"(scene_id={frame.scene_id!r}, image {tuple(frame.image.shape)})",
        flush=True,
    )
    # Lossless reference frame: diff against the same frame in
    # observations.mp4 to separate codec artifacts from render quality.
    PILImage.fromarray(np.asarray(frame.image)).save(out_dir / "first_frame.png")

    print(
        "[house-viz] building JAX VGGT extractor "
        "(loads weights; first extract is slow)...",
        flush=True,
    )
    extractor = JAXVGGTFeatureExtractor(
        total_budget=VGGTEncoder.VGGT_TOTAL_BUDGET,
        budgets_static=VGGTEncoder.VGGT_STATIC_BUDGETS,
        compute_heads=True,
        reset_mode=ResetMode.PERSIST_SCENE,
    )

    agent = RandomAgent(env, seed=args.seed)
    video_frames: list[np.ndarray] = []
    # Host-side accumulation: each step's kept points land as float32/uint8
    # NumPy immediately, so device memory stays bounded over long runs
    # (episodes x steps x 518^2 device-resident points would not fit in HBM).
    xyz_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []
    raw_points = 0
    total_frames = args.episodes * args.steps
    frame_no = 0
    t0 = time.perf_counter()
    for episode in range(args.episodes):
        if episode > 0:
            # Same-scene episode boundary: only env.reset() here. The
            # extractor's scene-aware reset fires inside extract() on the
            # is_first frame (reset_for_scene), and under PERSIST_SCENE a
            # same-scene reset is a no-op — the VGGT attention stream and
            # hence the shared world frame carry over, mirroring
            # evaluate.py's _start_eval_episode. Never call
            # extractor.reset() manually: that would wipe the cache and
            # re-anchor the world frame.
            frame = env.reset()
        episode_start_points = raw_points
        for step in range(args.steps):
            frame_no += 1
            t_frame = time.perf_counter()
            features = extractor.extract(frame)
            extract_s = time.perf_counter() - t_frame
            if args.video_fps > 0:
                # Copy to host immediately: 500 device-resident 518^2 frames
                # would hold ~400 MB of HBM alongside the VGGT KV cache.
                video_frames.append(np.asarray(frame.image))

            xyz_flat, rgb_flat = flattened_xyz_rgb(
                features.world_points, frame.image
            )
            # Production parity (house_context_pose_buffer): keep only finite,
            # confident points before accumulating.
            conf = features.confidence.reshape(-1)
            keep = (
                jnp.isfinite(xyz_flat).all(axis=1)
                & jnp.isfinite(conf)
                & (conf >= args.confidence_score)
            )
            xyz_np, rgb_np = xyz_flat[keep], rgb_flat[keep]
            xyz_parts.append(xyz_np)
            rgb_parts.append(rgb_np)
            raw_points += xyz_np.shape[0]

            print(
                f"[house-viz]   frame {frame_no}/{total_frames} "
                f"(episode {episode + 1}, step {step + 1}/{args.steps}): "
                f"extract {extract_s:.2f}s, {raw_points} raw points accumulated "
                f"(world_points {tuple(features.world_points.shape)})",
                flush=True,
            )
            if frame_no % 50 == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"[house-viz] progress {frame_no}/{total_frames}: "
                    f"{elapsed:.0f}s elapsed, {elapsed / frame_no:.2f}s/frame",
                    flush=True,
                )
            frame = agent.act()
        print(
            f"[house-viz] episode {episode + 1}/{args.episodes} done: "
            f"{raw_points - episode_start_points} points kept this episode, "
            f"{raw_points} total",
            flush=True,
        )
    t_concat = time.perf_counter()
    # Host concatenation, then one transfer to the device for the JAX
    # downsample. float32 (not bf16): coarser resolution would corrupt the
    # voxel assignment, see voxel_down_sample.
    xyz_all = jnp.asarray(np.concatenate(xyz_parts, axis=0))
    rgb_all = (
        jnp.asarray(np.concatenate(rgb_parts, axis=0), dtype=jnp.float32) / 255.0
    )
    print(
        f"[house-viz] concatenated {len(xyz_parts)} frames -> "
        f"{xyz_all.shape[0]} raw points in {time.perf_counter() - t_concat:.2f}s",
        flush=True,
    )
    t_down = time.perf_counter()
    # VGGT coordinates are scene-scale-normalized (metric scale is
    # unobservable from RGB), so voxel sizes are VGGT units, not meters.
    # Default: size the voxel from the cloud extent so map resolution is
    # scene-independent; --down-voxel pins an absolute edge instead.
    extent = jnp.ptp(xyz_all, axis=0)
    down_voxel = (
        args.down_voxel
        if args.down_voxel is not None
        else float(extent.max()) / args.down_cells
    )
    print(
        f"[house-viz] cloud extent (VGGT units): "
        f"{float(extent[0]):.2f} x {float(extent[1]):.2f} x "
        f"{float(extent[2]):.2f}; down_voxel={down_voxel:.4f} units "
        + (
            "(--down-voxel override)"
            if args.down_voxel is not None
            else f"(max extent / {args.down_cells} cells)"
        ),
        flush=True,
    )
    # Downsample on the GPU in JAX (Open3D's CUDA voxel_down_sample segfaults
    # on this cluster, see src.shared.pointcloud docstring); the PLY write is
    # pure NumPy (no Open3D anywhere in this script).
    xyz_down, rgb_down = voxel_down_sample(xyz_all, rgb_all, down_voxel)
    kept = xyz_down.shape[0]
    final_path = out_dir / "final.ply"
    print(
        f"[house-viz] voxel_down_sample({down_voxel:g} units): "
        f"{xyz_all.shape[0]} -> {kept} points "
        f"({kept / max(xyz_all.shape[0], 1) * 100:.1f}% kept, "
        f"{time.perf_counter() - t_down:.2f}s)",
        flush=True,
    )
    print(
        f"[house-viz] writing final cloud: {kept} points -> {final_path}",
        flush=True,
    )
    write_world_points_ply(xyz=xyz_down, rgb=rgb_down, path=final_path)
    if video_frames:
        t_video = time.perf_counter()
        video_path = write_frames_mp4(
            video_frames, out_dir / "observations.mp4", fps=args.video_fps
        )
        print(
            f"[house-viz] observation video: {video_path} "
            f"({len(video_frames)} frames @ {args.video_fps} fps, "
            f"{time.perf_counter() - t_video:.1f}s)",
            flush=True,
        )
    print(
        f"[house-viz] done in {time.perf_counter() - t0:.0f}s -> {out_dir}",
        flush=True,
    )


def main() -> None:
    """Run the collection and record ok/failed status in MANIFEST.json."""
    args = parse_args()
    print(f"jax backend: {jax.default_backend()}  devices: {jax.devices()}")
    print(
        f"[house-viz] config: steps={args.steps} episodes={args.episodes} "
        f"seed={args.seed} curriculum={args.curriculum} "
        f"conf>={args.confidence_score:g} "
        + (
            f"down_voxel={args.down_voxel:g} (override)"
            if args.down_voxel is not None
            else f"down_cells={args.down_cells}"
        ),
        flush=True,
    )

    run_tag = (
        time.strftime("%Y%m%d_%H%M%S")
        + f"_seed{args.seed}_ep{args.episodes}_steps{args.steps}"
    )
    out_dir = Path(args.output) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[house-viz] output dir: {out_dir}", flush=True)

    # Habitat GL teardown poisons exit codes on this cluster: MANIFEST.json
    # status is the ground truth for whether the run succeeded.
    write_manifest_start(out_dir, vars(args))
    try:
        collect_and_export(args, out_dir)
    except BaseException:
        write_manifest_end(out_dir, "failed")
        raise
    write_manifest_end(out_dir, "ok")


if __name__ == "__main__":
    main()
