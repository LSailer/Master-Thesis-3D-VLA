"""Generate 4 curriculum config JSON files for HM3D ObjectNav training.

Loads episodes via Habitat's dataset API (which assigns globally-unique
episode IDs), filters by scene + object category, splits 90/10 into
train/eval (seed 42), and writes config JSONs.
"""

import argparse
import json
from pathlib import Path

import numpy as np

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


def load_habitat_dataset():
    """Load all episodes via Habitat's API (globally-unique IDs)."""
    import habitat

    hab_cfg = habitat.get_config(
        "benchmark/nav/objectnav/objectnav_hm3d.yaml"
    )
    from omegaconf import OmegaConf
    with habitat.config.read_write(hab_cfg):
        hab_cfg.habitat.dataset.split = "train"
        hab_cfg.habitat.dataset.data_path = (
            "data/datasets/objectnav/hm3d/objectnav_hm3d_v2/{split}/{split}.json.gz"
        )
    dataset = habitat.make_dataset(
        id_dataset=hab_cfg.habitat.dataset.type,
        config=hab_cfg.habitat.dataset,
    )
    print(f"Loaded {len(dataset.episodes)} episodes via Habitat API")
    return dataset.episodes


def generate_level(level: dict, all_episodes: list, output_dir: Path) -> None:
    """Generate a single curriculum level config file."""
    name = level["name"]
    scenes_list = level["scenes"]
    scenes = set(scenes_list)
    categories = set(level["categories"])

    # Composite keys: [episode_id, object_category, scene_name]
    # Episode IDs are Habitat's globally-unique IDs
    episode_keys = []
    for ep in all_episodes:
        scene_name = ep.scene_id.split("/")[-1].replace(".basis.glb", "")
        if scene_name in scenes and ep.object_category in categories:
            episode_keys.append([ep.episode_id, ep.object_category, scene_name])

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
        "scenes": scenes_list,
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
            "scenes_count": len(scenes_list),
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

    print("Loading Habitat dataset...")
    all_episodes = load_habitat_dataset()

    print("\nGenerating curriculum configs...\n")
    for level in LEVELS:
        generate_level(level, all_episodes, args.output_dir)
        print()
    print("Done.")


if __name__ == "__main__":
    main()
