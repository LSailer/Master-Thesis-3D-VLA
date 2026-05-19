"""Build a held-out-house L3 chair-only curriculum for ObjectNav eval.

The generated curriculum uses HM3D train-split scenes that are disjoint from
the existing L3 10-house curriculum. It keeps the same key format consumed by
``HabitatObjectNavEnv``: ``[episode_id, object_category, scene_name]``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path


def _scene_name(path: Path) -> str:
    return path.name.replace(".json.gz", "")


def _load_scene_episodes(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return json.load(f).get("episodes", [])


def _episode_key(ep: dict, scene: str) -> list[str]:
    return [str(ep["episode_id"]), ep["object_category"], scene]


def build_curriculum(
    *,
    source_curriculum: Path,
    content_dir: Path,
    output: Path,
    houses: int,
    eval_episodes_per_house: int,
    seed: int,
) -> dict:
    source = json.loads(source_curriculum.read_text())
    source_scenes = set(source["scenes"])
    candidates: list[tuple[str, list[dict]]] = []

    for path in sorted(content_dir.glob("*.json.gz")):
        scene = _scene_name(path)
        if scene in source_scenes:
            continue
        chair_eps = [
            ep for ep in _load_scene_episodes(path)
            if ep.get("object_category") == "chair"
        ]
        if chair_eps:
            candidates.append((scene, chair_eps))

    if len(candidates) < houses:
        raise RuntimeError(
            f"Only found {len(candidates)} held-out chair scenes, need {houses}"
        )

    candidates.sort(key=lambda item: (-len(item[1]), item[0]))
    selected = candidates[:houses]
    rng = random.Random(seed)

    train_keys: list[list[str]] = []
    eval_keys: list[list[str]] = []
    stats: dict[str, dict[str, int]] = {}
    for scene, episodes in selected:
        shuffled = list(episodes)
        rng.shuffle(shuffled)
        eval_eps = shuffled[:eval_episodes_per_house]
        train_eps = shuffled[eval_episodes_per_house:]
        eval_keys.extend(_episode_key(ep, scene) for ep in eval_eps)
        train_keys.extend(_episode_key(ep, scene) for ep in train_eps)
        stats[scene] = {
            "chair_episodes": len(episodes),
            "eval_episodes": len(eval_eps),
            "unused_train_keys": len(train_eps),
        }

    result = {
        "name": "level3_heldout_10houses_1goal",
        "description": (
            "Held-out-house L3 ObjectNav eval curriculum: chair episodes from "
            "10 HM3D train-split scenes disjoint from level3_10houses_1goal."
        ),
        "source_curriculum": str(source_curriculum),
        "heldout_from_scenes": sorted(source_scenes),
        "scenes": [scene for scene, _episodes in selected],
        "categories": ["chair"],
        "seed": seed,
        "train_ratio": None,
        "eval_sample_size": len(eval_keys),
        "train_episode_keys": train_keys,
        "eval_episode_keys": eval_keys,
        "stats": stats,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_curriculum",
        type=Path,
        default=Path("data/curriculum/level3_10houses_1goal.json"),
    )
    parser.add_argument(
        "--content_dir",
        type=Path,
        default=Path("data/datasets/objectnav/hm3d/objectnav_hm3d_v2/train/content"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/curriculum/level3_heldout_10houses_1goal.json"),
    )
    parser.add_argument("--houses", type=int, default=10)
    parser.add_argument("--eval_episodes_per_house", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    curriculum = build_curriculum(
        source_curriculum=args.source_curriculum,
        content_dir=args.content_dir,
        output=args.output,
        houses=args.houses,
        eval_episodes_per_house=args.eval_episodes_per_house,
        seed=args.seed,
    )
    print(f"Wrote {args.output}")
    print(f"Held-out scenes: {', '.join(curriculum['scenes'])}")
    print(f"Eval episodes: {curriculum['eval_sample_size']}")


if __name__ == "__main__":
    main()
