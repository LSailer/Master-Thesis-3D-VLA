"""Diagnose whether streamed VGGT world_points stay anchored to frame 0.

Streams a short random walk and records, per frame, (a) the confidence-
filtered centroid of the predicted point map and (b) the Habitat agent's
ground-truth position. If VGGT's frame-0 anchor holds over the episode,
the centroid trajectory should sweep through space roughly like the agent
does; if the bounded KV budget degrades long-range anchoring, centroids
stay near the origin while the agent walks meters.

Run on a GPU node:
    .venv/bin/python prototyp/live_vggt/diag_frame_anchoring.py --steps 150
"""

from __future__ import annotations

import argparse

import jax.numpy as jnp
import numpy as np

from src.baselines.random_agent import RandomAgent
from src.r2dreamer.encoders.base import VGGTEncoder
from src.r2dreamer.launch.habitat_setup import make_habitat_env
from src.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor, ResetMode


def main() -> None:
    """Stream frames and report centroid-vs-agent trajectory statistics."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--confidence-score", type=float, default=1.5)
    p.add_argument("--out", type=str, default="outputs/prototype/live_vggt/diag_anchoring.npz")
    args = p.parse_args()

    env = make_habitat_env(curriculum="L1", render_resolution=518, seed=args.seed)
    frame = env.reset()
    extractor = JAXVGGTFeatureExtractor(
        total_budget=VGGTEncoder.VGGT_TOTAL_BUDGET,
        budgets_static=VGGTEncoder.VGGT_STATIC_BUDGETS,
        compute_heads=True,
        reset_mode=ResetMode.PERSIST_SCENE,
    )
    agent = RandomAgent(env, seed=args.seed)

    centroids, agent_pos = [], []
    for step in range(args.steps):
        features = extractor.extract(frame)
        xyz = features.world_points.reshape(-1, 3).astype(jnp.float32)
        conf = features.confidence.reshape(-1)
        keep = jnp.isfinite(xyz).all(axis=1) & (conf >= args.confidence_score)
        n_keep = int(keep.sum())
        if n_keep > 0:
            centroid = jnp.where(keep[:, None], xyz, 0.0).sum(axis=0) / n_keep
            centroids.append(np.asarray(centroid))
        else:
            centroids.append(np.full(3, np.nan))
        agent_pos.append(np.asarray(env.agent_state.position, dtype=np.float32))
        if (step + 1) % 25 == 0:
            print(f"[diag] {step + 1}/{args.steps}", flush=True)
        frame = agent.act()

    cent = np.stack(centroids)
    apos = np.stack(agent_pos)
    np.savez(args.out, centroids=cent, agent_positions=apos)

    def extent(x: np.ndarray) -> np.ndarray:
        return np.nanmax(x, axis=0) - np.nanmin(x, axis=0)

    cent_step = np.linalg.norm(np.diff(cent, axis=0), axis=1)
    apos_step = np.linalg.norm(np.diff(apos, axis=0), axis=1)
    moved = apos_step > 1e-4  # only steps where the agent actually translated
    print(f"[diag] agent path extent (m):    {np.round(extent(apos), 2)}", flush=True)
    print(f"[diag] centroid extent:          {np.round(extent(cent), 2)}", flush=True)
    print(
        f"[diag] agent total path length:  {apos_step.sum():.1f} m over "
        f"{int(moved.sum())} moving steps",
        flush=True,
    )
    print(f"[diag] centroid total drift:     {cent_step.sum():.1f}", flush=True)
    if moved.sum() > 2:
        r = np.corrcoef(apos_step[moved], cent_step[moved])[0, 1]
        print(f"[diag] step-size correlation (moving steps only): {r:.3f}", flush=True)
    print(f"[diag] saved {args.out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
