"""Pre-render data bundle for the debug-viz notebook.

Loads sparse frames + agent trajectories from the per-step npz dumps and saves
a compact npz that the notebook can load fast (no need to reopen 635 npz files
when the notebook is run). Also stages the summary stats.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _load_sparse(ep_dir: Path, frame_indices: list[int]):
    """Load only the requested frames; also load full agent trajectory."""
    steps = sorted(p for p in ep_dir.iterdir() if p.name.startswith("step_") and p.suffix == ".npz")
    T = len(steps)
    # full trajectory + actions at every step (cheap)
    positions = np.zeros((T, 3), dtype=np.float32)
    rotations = np.zeros((T, 4), dtype=np.float32)
    rewards = np.zeros((T,), dtype=np.float32)
    actions = np.zeros((T,), dtype=np.int32)
    for i, s in enumerate(steps):
        d = np.load(s)
        positions[i] = d["agent_position"]
        rotations[i] = d["agent_rotation"]
        rewards[i] = d["reward"]
        actions[i] = d["action"]
    # sparse heavy frames
    chosen_world_points = []
    chosen_rgb = []
    for fi in frame_indices:
        d = np.load(steps[fi])
        chosen_world_points.append(d["world_points"])
        chosen_rgb.append(d["rgb"])
    return {
        "T": T,
        "trajectory": positions,
        "rotations": rotations,
        "rewards": rewards,
        "actions": actions,
        "chosen_frame_indices": np.array(frame_indices, dtype=np.int32),
        "chosen_world_points": np.stack(chosen_world_points),  # (K, 37, 37, 3)
        "chosen_rgb": np.stack(chosen_rgb),                     # (K, 3, 518, 518)
    }


def main() -> None:
    dump_dir = Path("output/runs/r2dreamer-curriculum-l1-vggt/baseline-actent3e-4/debug/viz-pair-a")
    probe_dir = Path("output/methods/debug_viz/l1/probes")
    sim_dir = Path("output/methods/debug_viz/l1/similarity")
    out_dir = Path("output/methods/debug_viz/l1/notebook")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sparse-frame strategy: load 5 frames per episode (start, 25%, 50%, 75%, end-1)
    eval_results = json.loads((dump_dir / "eval_results.json").read_text())["results"]
    info = {r["episode"]: r for r in eval_results}

    bundles = {}
    for ep_idx in (1, 7):
        ep_dir = dump_dir / f"episode_{ep_idx:03d}"
        T = len(list(ep_dir.glob("step_*.npz")))
        frames = [0, T // 4, T // 2, (3 * T) // 4, T - 1]
        print(f"ep{ep_idx}: T={T}, sampling frames {frames}")
        bundle = _load_sparse(ep_dir, frames)
        bundle["meta"] = {
            "episode_idx": ep_idx,
            "success": int(info[ep_idx]["success"]),
            "spl": float(info[ep_idx]["spl"]),
            "steps": int(info[ep_idx]["steps"]),
            "reward": float(info[ep_idx]["reward"]),
            "start_position": list(info[ep_idx]["start_position"]),
            "goal_positions": [list(g) for g in info[ep_idx]["goal_positions"]],
        }
        # Also load probe predictions + similarity (much smaller)
        pred = np.load(probe_dir / f"predictions_ep{ep_idx:03d}.npz")
        sim = np.load(sim_dir / f"ep_{ep_idx:03d}/similarity.npz")
        bundle["pred_world_points_feat"] = pred["world_points_pred_feat"][frames]
        bundle["pred_world_points_deter"] = pred["world_points_pred_deter"][frames]
        bundle["pred_world_points_stoch"] = pred["world_points_pred_stoch"][frames]
        bundle["mse_feat"] = pred["mse_feat"]
        bundle["mse_deter"] = pred["mse_deter"]
        bundle["mse_stoch"] = pred["mse_stoch"]
        bundle["S_VGGT"] = sim["S_VGGT"]
        bundle["S_feat"] = sim["S_feat"]
        bundle["S_deter"] = sim["S_deter"]
        bundle["diff_feat"] = sim["diff_feat"]
        bundle["diff_deter"] = sim["diff_deter"]
        bundles[ep_idx] = bundle

    # Save flat npz per episode (notebook does np.load and accesses keys)
    for ep_idx, b in bundles.items():
        flat = {k: v for k, v in b.items() if k not in ("meta",)}
        flat["meta_json"] = np.array(json.dumps(b["meta"]))
        np.savez_compressed(out_dir / f"bundle_ep{ep_idx:03d}.npz", **flat)
        print(f"  wrote {out_dir / f'bundle_ep{ep_idx:03d}.npz'}: {sum(v.nbytes for v in flat.values() if hasattr(v, 'nbytes')) / 1e6:.1f} MB raw")

    print(f"\nDone. Bundles in {out_dir}/")


if __name__ == "__main__":
    main()
