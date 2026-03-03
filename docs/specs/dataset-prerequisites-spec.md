# PR Spec: Dataset & Environment Prerequisites

## Context

This is the prerequisite work needed before implementing the action heads spec (`action-heads-spec.md` on `main`). Everything here happens on `feat/datasets`.

## Current State

| Component | Status | Evidence |
|-----------|--------|----------|
| habitat-sim | Installed (v0.3.3 via Pixi) | `pixi.lock` |
| habitat-lab | Cloned | `external/habitat-lab/` |
| Episode data | Ready (145 train scenes, ~7.25M episodes) | `data/datasets/objectnav/hm3d/objectnav_hm3d_v2/train/content/*.json.gz` |
| Episode statistics | Done | `notebooks/habitat_objectnav_benchmark.ipynb` (sections 1-5) |
| HM3D scene meshes (.glb) | **MISSING** | Notebook cell 22: "No .glb scene files found" |
| Habitat env test | **SKIPPED** | Notebook cells 27-29 all skipped (no scenes) |
| ShortestPathFollower test | **NOT DONE** | No code anywhere |
| Expert demo collection | **NOT DONE** | No script |
| Expert dataset on disk | **NOT DONE** | Nothing saved |

## Blocker: HM3D Scene Meshes

Everything depends on downloading the scene meshes. Without `.glb` files Habitat can't render observations.

### How to download

Requires a free Matterport API token:
1. Register at https://matterport.com/partners/facebook (free for academic use)
2. Get `API_TOKEN_ID` and `API_TOKEN_SECRET`
3. Run:

```bash
# Start with minival (2 scenes, ~500 MB) to verify pipeline
python -m habitat_sim.utils.datasets_download \
    --username <API_TOKEN_ID> --password <API_TOKEN_SECRET> \
    --data-path data/ \
    --uids hm3d_minival_v0.2

# Then full val (36 scenes, ~5 GB)
python -m habitat_sim.utils.datasets_download \
    --username <API_TOKEN_ID> --password <API_TOKEN_SECRET> \
    --data-path data/ \
    --uids hm3d_val_v0.2

# Then train (145 scenes, ~25 GB) — only when ready to scale
python -m habitat_sim.utils.datasets_download \
    --username <API_TOKEN_ID> --password <API_TOKEN_SECRET> \
    --data-path data/ \
    --uids hm3d_train_v0.2
```

Expected result: `data/scene_datasets/hm3d_v0.2/{minival,val,train}/00xxx-SceneID/SceneID.basis.glb`

## Tasks (in order)

### Task 1: Download HM3D minival scenes

- Get Matterport API token
- Download `hm3d_minival_v0.2` (2 scenes, ~500 MB)
- Verify `.glb` + `.navmesh` files exist under `data/scene_datasets/hm3d_v0.2/minival/`

### Task 2: Verify Habitat env runs

Complete notebook sections 6 (cells 27-29):
- `import habitat` + `import habitat_sim` succeeds
- Load ObjectNav config for `val_mini` split
- `env.reset()` returns obs with `rgb` (256x256x3) and `depth` (256x256x1)
- Take a few actions, visualize RGB+depth frames
- Confirm episode info contains geodesic_distance

### Task 3: Test ShortestPathFollower

Add a new notebook section or extend section 6:

```python
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

follower = ShortestPathFollower(env.sim, goal_radius=0.2, return_one_hot=False)

obs = env.reset()
goal = env.current_episode.goals[0].position
actions_taken = []

while not env.episode_over:
    action = follower.get_next_action(goal)
    actions_taken.append(action)
    obs = env.step(action)

print(f"Episode done in {len(actions_taken)} steps")
print(f"Actions: {Counter(actions_taken)}")
print(f"Success: {env.get_metrics().get('success', '?')}")
```

Verify:
- [ ] Returns valid actions (0-3)
- [ ] Episode completes successfully (agent reaches goal)
- [ ] Action distribution is reasonable (not all same action)
- [ ] SPL ≈ 1.0 for oracle follower

### Task 4: Write expert demo collection script

Create `scripts/collect_expert_demos.py`:

```
Usage: pixi run python scripts/collect_expert_demos.py \
    --split val_mini \
    --n-episodes 100 \
    --output data/expert_demos/val_mini_100.pt
```

For each episode, save:
```python
{
    "episode_id": str,
    "scene_id": str,
    "object_category": str,
    "observations": [{
        "rgb": np.uint8 [H, W, 3],     # or skip if too large, re-render later
        "depth": np.float32 [H, W, 1],
    }],
    "actions": [int],                    # list of expert actions per step
    "success": bool,
    "spl": float,
}
```

Storage considerations:
- RGB at 256x256x3 uint8 = 192 KB/frame
- 100 steps avg × 100 episodes = 10K frames ≈ 1.9 GB with RGB
- Option: save only actions + metadata, re-render obs at train time (Habitat is fast enough)
- Start with **actions-only** format (~few MB), add obs later if needed

### Task 5: Collect initial expert dataset

Run collection script:
```bash
# Small sanity set (minutes)
pixi run python scripts/collect_expert_demos.py --split val_mini --n-episodes 30

# Medium set for first experiments (30 min - 1 hr)
pixi run python scripts/collect_expert_demos.py --split val --n-episodes 1000
```

Verify:
- [ ] Dataset loads correctly
- [ ] Action distribution: ~40% FORWARD, ~25% LEFT, ~25% RIGHT, ~10% STOP (rough)
- [ ] Average episode length: 30-100 steps
- [ ] Oracle SPL ≈ 1.0 for all episodes

### Task 6: Update notebook with env test results

Fill in notebook sections 6 (cells 27-29) with actual outputs:
- Habitat version info
- RGB + depth visualization from a real episode
- ShortestPathFollower walkthrough

## File Structure (new files this PR)

```
scripts/
└── collect_expert_demos.py    # Expert data collection

data/
├── scene_datasets/
│   └── hm3d_v0.2/            # Downloaded scene meshes (gitignored)
│       ├── minival/
│       ├── val/
│       └── train/
└── expert_demos/              # Collected expert demonstrations (gitignored)
    ├── val_mini_30.pt
    └── val_1000.pt

docs/specs/
└── dataset-prerequisites-spec.md   # This file
```

## .gitignore additions

```
data/scene_datasets/hm3d_v0.2/
data/expert_demos/
```

## Dependencies

No new deps — habitat-sim and habitat-lab already available via Pixi.

## Done criteria

```
[ ] HM3D minival scenes downloaded (.glb + .navmesh)
[ ] Habitat env loads and renders observations
[ ] ShortestPathFollower completes episodes with SPL ≈ 1.0
[ ] collect_expert_demos.py works end-to-end
[ ] 30 val_mini expert demos saved to disk
[ ] Notebook updated with env test outputs
───── ready for action-heads-spec on new branch ─────
```
