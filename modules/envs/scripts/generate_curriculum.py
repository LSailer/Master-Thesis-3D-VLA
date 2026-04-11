"""Generate 4 curriculum config JSON files for HM3D ObjectNav training.

Loads episodes from scene .json.gz files, filters by object category,
splits 90/10 into train/eval (seed 42), and writes config JSONs.
"""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

DATASET_DIR = Path("data/datasets/objectnav/hm3d/objectnav_hm3d_v2/train/content")

ALL_CATEGORIES = ["bed", "chair", "plant", "sofa", "toilet", "tv_monitor"]

SCENES_10 = [
    "fK2vEV32Lag",  # easy (shared with L1-2)
    "W9YAR9qcuvN",  # easy
    "wPLokgvCnuk",  # easy
    "ACZZiU6BXLz",  # easy
    "XfUxBGTFQQb",  # medium
    "9h5JJxM6E5S",  # medium
    "qz3829g1Lzf",  # medium
    "oPj9qMxrDEa",  # medium
    "u5atqC7vRCY",  # hard
    "j2EJhFEQGCL",  # hard
]

LEVELS = [
    {
        "name": "level1_1house_1goal",
        "description": "Single house (fK2vEV32Lag), chair only — proves WM learns",
        "scenes": ["fK2vEV32Lag"],
        "categories": ["chair"],
    },
    {
        "name": "level2_1house_6goals",
        "description": "Single house (fK2vEV32Lag), all 6 categories — tests multi-goal",
        "scenes": ["fK2vEV32Lag"],
        "categories": ALL_CATEGORIES,
    },
    {
        "name": "level3_10houses_1goal",
        "description": "10 houses (easy/medium/hard), chair only — tests generalization",
        "scenes": SCENES_10,
        "categories": ["chair"],
    },
    {
        "name": "level4_10houses_6goals",
        "description": "10 houses (easy/medium/hard), all 6 categories — full curriculum",
        "scenes": SCENES_10,
        "categories": ALL_CATEGORIES,
    },
]

SEED = 42
TRAIN_RATIO = 0.9
EVAL_SAMPLE_SIZE = 50


def load_episodes(scene: str) -> list[dict]:
    """Load all episodes from a scene's .json.gz file."""
    path = DATASET_DIR / f"{scene}.json.gz"
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    return data["episodes"]


def generate_level(level: dict, output_dir: Path) -> None:
    """Generate a single curriculum level config file."""
    name = level["name"]
    scenes = level["scenes"]
    categories = set(level["categories"])

    # Composite keys: [episode_id, object_category, scene_name]
    # IDs repeat across both categories and scenes
    episode_keys = []
    for scene in scenes:
        for ep in load_episodes(scene):
            if ep["object_category"] in categories:
                episode_keys.append([ep["episode_id"], ep["object_category"], scene])

    # Shuffle and split
    rng = np.random.RandomState(SEED)
    indices = np.arange(len(episode_keys))
    rng.shuffle(indices)

    split = int(len(indices) * TRAIN_RATIO)
    train_keys = [episode_keys[i] for i in indices[:split]]
    eval_keys = [episode_keys[i] for i in indices[split:]]

    config = {
        "name": name,
        "description": level["description"],
        "scenes": scenes,
        "categories": level["categories"],
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "eval_sample_size": EVAL_SAMPLE_SIZE,
        "train_episode_keys": train_keys,
        "eval_episode_keys": eval_keys,
        "stats": {
            "total_episodes": len(episode_keys),
            "train_episodes": len(train_keys),
            "eval_episodes": len(eval_keys),
            "scenes_count": len(scenes),
            "categories_count": len(level["categories"]),
        },
    }

    out_path = output_dir / f"{name}.json"
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)

    stats = config["stats"]
    print(f"  {name}:")
    print(f"    scenes={stats['scenes_count']}, categories={stats['categories_count']}")
    print(f"    total={stats['total_episodes']}, train={stats['train_episodes']}, eval={stats['eval_episodes']}")
    print(f"    wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate HM3D ObjectNav curriculum configs")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/curriculum"),
        help="Output directory for curriculum JSON files (default: data/curriculum/)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating curriculum configs...\n")
    for level in LEVELS:
        generate_level(level, args.output_dir)
        print()
    print("Done.")


if __name__ == "__main__":
    main()
