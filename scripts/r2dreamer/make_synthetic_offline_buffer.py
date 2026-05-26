"""Generate a tiny synthetic offline buffer for smoke-testing 3D-26.

Mirrors the on-disk layout produced by collect_offline_buffer.py — same file
names, same keys, same dtypes — but with random data and a small step count.
Lets us exercise the offline training pipeline on a dev GPU before the real
400k collection job finishes.

Usage:
    python scripts/r2dreamer/make_synthetic_offline_buffer.py \\
        --out-dir data/offline_buffer_smoke --n-steps 3000 --n-episodes 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

WP_CP_DIM = 4116
AGGREGATOR_DIM = 3072


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n-steps", type=int, default=3000)
    p.add_argument("--n-episodes", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    actions = rng.integers(0, 4, size=args.n_steps).astype(np.int32)
    rewards = rng.standard_normal(args.n_steps).astype(np.float32) * 0.5
    dones = np.zeros(args.n_steps, dtype=np.bool_)
    episode_ids = np.zeros(args.n_steps, dtype=np.int32)

    boundaries = np.linspace(0, args.n_steps, args.n_episodes + 1, dtype=np.int32)
    for ep in range(args.n_episodes):
        start, end = int(boundaries[ep]), int(boundaries[ep + 1])
        episode_ids[start:end] = ep
        dones[end - 1] = True

    np.savez(
        out_dir / "trajectory_skeleton.npz",
        action=actions, reward=rewards, done=dones, episode_id=episode_ids,
    )

    z_wp = rng.standard_normal((args.n_steps, WP_CP_DIM)).astype(np.float16) * 0.1
    z_agg = rng.standard_normal((args.n_steps, AGGREGATOR_DIM)).astype(np.float16) * 0.1
    np.savez(out_dir / "z_wp_cp.npz", features=z_wp)
    np.savez(out_dir / "z_aggregator.npz", features=z_agg)

    heldout_episodes = max(1, args.n_episodes // 10)
    heldout_start = args.n_episodes - heldout_episodes
    metadata = {
        "issue": "3D-26-smoke",
        "status": "completed",
        "n_completed_steps": args.n_steps,
        "num_episodes": args.n_episodes,
        "heldout_split": {
            "rule": "last_10_percent_of_episodes",
            "episode_id_start_inclusive": heldout_start,
            "episode_id_end_exclusive": args.n_episodes,
            "num_episodes": heldout_episodes,
        },
        "code_sha": "smoke",
        "checkpoint_sha256": "smoke",
        "collect_seed": args.seed,
        "synthetic": True,
    }
    (out_dir / "collection_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Wrote synthetic buffer to {out_dir} ({args.n_steps} steps, {args.n_episodes} episodes)")


if __name__ == "__main__":
    main()
