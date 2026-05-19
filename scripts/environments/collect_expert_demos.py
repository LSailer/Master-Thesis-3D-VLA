"""Collect expert demonstrations using Habitat ShortestPathFollower."""

import argparse
from pathlib import Path

import h5py
import tqdm

try:
    import habitat
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
except ImportError:
    raise ImportError("habitat-lab required: pip install habitat-lab habitat-sim")


def collect(config_path: str, output: str, max_episodes: int = 100) -> None:
    config = habitat.get_config(config_path)
    env = habitat.Env(config=config)
    follower = ShortestPathFollower(env.sim, goal_radius=0.2, return_one_hot=False)

    with h5py.File(output, "w") as f:
        ep_idx = 0
        for _ in tqdm.trange(max_episodes, desc="episodes"):
            obs = env.reset()
            grp = f.create_group(f"episode_{ep_idx}")
            step_idx = 0

            while not env.episode_over:
                action = follower.get_next_action(env.current_episode.goals[0].position)
                step_grp = grp.create_group(f"step_{step_idx}")
                step_grp.create_dataset("rgb", data=obs["rgb"], compression="gzip")
                step_grp.create_dataset("depth", data=obs["depth"], compression="gzip")
                step_grp.attrs["action"] = int(action)
                step_grp.attrs["goal_category"] = env.current_episode.object_category

                obs = env.step(action)
                step_idx += 1

            grp.attrs["n_steps"] = step_idx
            ep_idx += 1

    print(f"Saved {ep_idx} episodes to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="benchmark/nav/objectnav/objectnav_hm3d.yaml"
    )
    parser.add_argument("--output", default="data/expert_demos.h5")
    parser.add_argument("--max-episodes", type=int, default=100)
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    collect(args.config, args.output, args.max_episodes)
