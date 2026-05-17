# Curriculum Preprocessing Plan

**Branch:** `feature/baseline-training`
**Date:** 2026-04-11
**Status:** Planned

## Goal

Create 4 curriculum levels of increasing complexity for HM3D ObjectNav training. Each level isolates one variable (goal diversity or scene diversity), building a thesis narrative from "the world model learns" to "3D features help generalization."

## Curriculum Levels

| Level | Name | Houses | Goals | Purpose |
|-------|------|--------|-------|---------|
| 1 | `level1_1house_1goal` | fK2vEV32Lag | chair | Prove WM learns (vs random agent) |
| 2 | `level2_1house_6goals` | fK2vEV32Lag | all 6 | Show impact of missing goal conditioning |
| 3 | `level3_10houses_1goal` | 10 scenes | chair | Show WM struggles across houses |
| 4 | `level4_10houses_6goals` | 10 scenes | all 6 | Full complexity, motivates 3D features |

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Single house | `fK2vEV32Lag` | All 6 categories, 2.95m mean chair geodesic (2nd easiest), 50k episodes |
| Goal category | Chair | Most common indoor object, consistent reward signal, proposed in #58 |
| Train/eval split | 90/10 random by episode ID, seed 42 | Same house(s) for train and eval — isolates learning from generalization |
| 10-scene selection | 4 easy, 4 medium, 2 hard | Spread across difficulty; includes fK2vEV32Lag as controlled comparison |
| Storage format | Filter configs with explicit episode IDs | ~5 MB per config, fully reproducible, no recomputation risk |
| Config loading | `curriculum_path` + `curriculum_mode` on env | Single source of truth, replaces other filters when used |
| Eval episodes | 50 per level | Matches baseline plan, ~25 min runtime, enough for confidence intervals |
| Scene analysis | `output/scene_analysis/train_scenes_by_difficulty.json` | All 145 scenes ranked, with per-category stats, for thesis reference |

## Selected Scenes (Levels 3-4)

Source: `output/scene_analysis/train_scenes_by_difficulty.json`

| Tier | Rank | Scene | Mean geodesic | Chair geodesic |
|------|------|-------|---------------|----------------|
| Easy | 5 | **fK2vEV32Lag** (shared with L1-2) | 3.67m | 2.95m |
| Easy | 17 | W9YAR9qcuvN | 4.16m | 4.97m |
| Easy | 34 | wPLokgvCnuk | 4.62m | 5.23m |
| Easy | 45 | ACZZiU6BXLz | 4.96m | 5.48m |
| Medium | 46 | XfUxBGTFQQb | 5.01m | 3.79m |
| Medium | 65 | 9h5JJxM6E5S | 5.44m | 3.06m |
| Medium | 81 | qz3829g1Lzf | 5.83m | 7.91m |
| Medium | 92 | oPj9qMxrDEa | 6.18m | 3.17m |
| Hard | 93 | u5atqC7vRCY | 6.19m | 4.20m |
| Hard | 145 | j2EJhFEQGCL | 13.39m | 8.27m |

## Deliverables

### 1. Curriculum generation script

**New file:** `modules/envs/scripts/generate_curriculum.py`

- Loads per-scene `.json.gz` files for selected scenes
- Filters by `object_category` where needed
- Splits episodes 90/10 (train/eval) with seed 42
- Writes 4 configs to `data/curriculum/`
- Deterministic, run once, commit output

### 2. Curriculum config files

**New files in** `data/curriculum/`:

```
level1_1house_1goal.json      ~1 MB (8,333 chair episodes from fK2vEV32Lag)
level2_1house_6goals.json     ~5 MB (50,000 episodes from fK2vEV32Lag)
level3_10houses_1goal.json    ~8 MB (~83,000 chair episodes from 10 scenes)
level4_10houses_6goals.json  ~50 MB (~500,000 episodes from 10 scenes)
```

Config format:

```json
{
  "name": "level1_1house_1goal",
  "description": "Single house (fK2vEV32Lag), chair only",
  "scenes": ["fK2vEV32Lag"],
  "categories": ["chair"],
  "seed": 42,
  "train_ratio": 0.9,
  "eval_sample_size": 50,
  "train_episode_ids": ["ep_001", ...],
  "eval_episode_ids": ["ep_800", ...]
}
```

### 3. Environment integration

**Modify:** `modules/envs/habitat.py` (`HabitatObjectNavEnv.__init__`)

Add two new parameters:
- `curriculum_path: str | None` -- path to a curriculum config JSON
- `curriculum_mode: str` -- `"train"` or `"eval"`

When `curriculum_path` is set:
- Always load HM3D `train` split (all curriculum episodes come from train scenes)
- Filter `self._env._dataset.episodes` to matching episode IDs from the config
- Ignore `max_geodesic` and `step_counts_path` (curriculum is the single source of truth)

### 4. Analysis notebooks

**Notebook 1:** `modules/r2dreamer/notebooks/curriculum_level1_wm_vs_random.ipynb`
- Level 1 trained agent vs random agent
- Plots: loss curves, action distribution, reward distribution, trajectory maps
- Verdict: does WM learn navigation dynamics?

**Notebook 2:** `modules/r2dreamer/notebooks/curriculum_level1_vs_level2.ipynb`
- Level 1 (chair only) vs Level 2 (6 goals), both trained
- Plots: reward comparison, action distribution, per-category performance (L2)
- Verdict: how much does missing goal conditioning hurt?

**Notebook 3:** `modules/r2dreamer/notebooks/curriculum_level1_vs_level3.ipynb`
- Level 1 (1 house) vs Level 3 (10 houses), both chair only
- Plots: aggregate reward, per-scene breakdown, trajectory quality by difficulty tier
- Verdict: where does scene diversity break the WM?

**Notebook 4:** 3D features comparison -- tracked as git issue (not implemented until 3D agent exists)

### 5. Git issue for 3D notebook

Create issue: "Notebook: 3D vs 2D feature comparison on curriculum Levels 3-4"
- Placeholder for when 3D (UNITE) agent is implemented
- Requirements: same eval episodes as L3/L4, same metrics, side-by-side plots
- Depends on: 3D agent implementation, curriculum configs

## Implementation Order

```
[1] generate_curriculum.py     -- script to create configs
[2] data/curriculum/*.json     -- run script, commit configs
[3] habitat.py integration     -- curriculum_path + curriculum_mode
[4] Git issue for 3D notebook  -- placeholder
[5] Notebook 1 (WM vs random)  -- after Level 1 training completes
[6] Notebook 2 (L1 vs L2)      -- after Level 2 training completes
[7] Notebook 3 (L1 vs L3)      -- after Level 3 training completes
```

Steps 1-4 can be implemented now. Steps 5-7 depend on training runs completing.

## Related

- Baseline training plan: `docs/baseline-training-plan.md`
- Scene analysis: `output/scene_analysis/train_scenes_by_difficulty.json`
- Issue #57: Scene diversity (shuffle + train_ratio)
- Issue #58: Single-object curriculum (chair only)
- Issue #49: Episode filtering by shortest path steps (superseded by curriculum configs for filtered runs)
