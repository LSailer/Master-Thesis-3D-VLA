# PR Spec: Dataset & Environment Prerequisites

## Context

This is the prerequisite work needed before implementing the action heads spec (`action-heads-spec.md` on `main`). Everything here happens on `feat/datasets`.

## Decision: Pixi as single package manager

habitat-sim is only distributed via conda (no PyPI wheel). Rather than juggling two package managers (Pixi for habitat + uv for Python deps), use **Pixi for everything**. Pixi supports `[pypi-dependencies]` for pip packages alongside conda packages.

```
Before (split):          After (unified):
  pixi → habitat-sim      pixi → habitat-sim (conda)
  uv   → torch, etc.             torch, einops, etc. (pypi-dependencies)
                                  master-thesis-3d-vla (editable install)
```

All commands use `pixi run` as universal prefix.

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
| Pixi pypi-dependencies | **NOT SET UP** | `pixi.toml` only has conda deps, torch etc. installed separately via uv |

## Blocker: HM3D Scene Meshes

Everything depends on downloading the scene meshes. Without `.glb` files Habitat can't render observations.

### How to download

Requires a free Matterport API token:
1. Register at https://matterport.com/partners/facebook (free for academic use)
2. Get `HM3D_TOKEN_ID` (public) and `HM3D_TOKEN_SECRET` (private)
3. Add to `~/.zshrc` or `.env`:
   ```bash
   export HM3D_TOKEN_ID="your-public-token"
   export HM3D_TOKEN_SECRET="your-private-token"
   ```
4. Run:

```bash
# Start with minival (2 scenes, ~500 MB) to verify pipeline
pixi run python -m habitat_sim.utils.datasets_download \
    --username "$HM3D_TOKEN_ID" --password "$HM3D_TOKEN_SECRET" \
    --data-path data/ \
    --uids hm3d_minival_v0.2

# Then full val (36 scenes, ~5 GB)
pixi run python -m habitat_sim.utils.datasets_download \
    --username "$HM3D_TOKEN_ID" --password "$HM3D_TOKEN_SECRET" \
    --data-path data/ \
    --uids hm3d_val_v0.2

# Then train (145 scenes, ~25 GB) — only when ready to scale
pixi run python -m habitat_sim.utils.datasets_download \
    --username "$HM3D_TOKEN_ID" --password "$HM3D_TOKEN_SECRET" \
    --data-path data/ \
    --uids hm3d_train_v0.2
```

Expected result: `data/scene_datasets/hm3d_v0.2/{minival,val,train}/00xxx-SceneID/SceneID.basis.glb`

## Tasks (in order)

### Task 0: Migrate to Pixi as single package manager

Update `pixi.toml` to include all pip deps under `[pypi-dependencies]`:

```toml
[pypi-dependencies]
torch = ">=2.7"
torchvision = ">=0.22"
einops = "*"
safetensors = "*"
huggingface_hub = "*"
pillow = "*"
tqdm = "*"
plotly = ">=6.5"
manim = ">=0.18.1"
pytest = "*"
master-thesis-3d-vla = { path = ".", editable = true }
```

Verify:
```bash
pixi install
pixi run python -c "
import habitat_sim
import torch
import einops
print(f'habitat-sim: {habitat_sim.__version__}')
print(f'torch: {torch.__version__}')
print('all deps resolved')
"
```

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
pixi.toml                             # Updated with [pypi-dependencies]

scripts/
└── collect_expert_demos.py            # Expert data collection

data/
├── scene_datasets/
│   └── hm3d_v0.2/                    # Downloaded scene meshes (gitignored)
│       ├── minival/
│       ├── val/
│       └── train/
└── expert_demos/                      # Collected expert demonstrations (gitignored)
    ├── val_mini_30.pt
    └── val_1000.pt

docs/specs/
└── dataset-prerequisites-spec.md      # This file
```

## .gitignore additions

```
data/scene_datasets/hm3d_v0.2/
data/expert_demos/
.env
```

## Dependencies

All managed through `pixi.toml`:
- **Conda**: habitat-sim, python, numpy, cmake
- **PyPI** (via `[pypi-dependencies]`): torch, torchvision, einops, safetensors, huggingface_hub, pillow, tqdm, plotly, manim, pytest, this project (editable)

## Done criteria

```
[ ] pixi.toml updated, `pixi install` resolves all deps
[ ] `pixi run python -c "import habitat_sim; import torch"` works
[ ] HM3D minival scenes downloaded (.glb + .navmesh)
[ ] Habitat env loads and renders observations
[ ] ShortestPathFollower completes episodes with SPL ≈ 1.0
[ ] collect_expert_demos.py works end-to-end
[ ] 30 val_mini expert demos saved to disk
[ ] Notebook updated with env test outputs
───── ready for action-heads-spec on new branch ─────
```
