---
run_path: output/runs/r2dreamer-curriculum-l2/_blessed/l2-1house-6goals
slurm_id: 3957713
wandb_id: flky9ybh
status: blessed
---

# L2 — 1 House, 6 Goals, 2.4M Steps

**Status**: completed
**Slides**: [curriculum-scaling.html](../../curriculum-scaling.html)
**Date**: 2026-04-16
**Tags**: #r2dreamer #habitat #l2 #curriculum #multi-goal #buffer-fix
**Wandb**: [r2d-L2-buffix-3957713](https://wandb.ai/sailer-luca-university-ulm/3d-vla-objectnav/runs/flky9ybh)
**SLURM Job ID**: 3957713

## Setup

R2-Dreamer trained for 2.4M environment steps on the L2 curriculum: single house (`fK2vEV32Lag`), **all 6 goal categories** (bed, chair, plant, sofa, toilet, tv_monitor), no goal conditioning. 64x64 RGB observations, geodesic-delta reward with step penalty (-0.01/step) and success bonus (10.0), 1000 max steps per episode.

**Hypothesis**: With 6 goal categories in the same house, the agent should learn above random for all categories, but performance may vary by goal difficulty. Without goal conditioning, the agent cannot know which object to navigate to — it must develop a general navigation policy.

## Changes

Compared to L1 rerun (75% SR, chair only):
- **6 goal categories** instead of 1 — same house, same architecture, same hyperparameters
- Episode distribution: 8,333 episodes per goal (uniform across categories)

## Configuration

Same as L1 rerun except:
| Parameter | Value |
|-----------|-------|
| Curriculum | level2_1house_6goals |
| Goals | bed, chair, plant, sofa, toilet, tv_monitor |

## Results

### Overall Metrics

| Metric | L2 (6 goals) | L1 Rerun (chair) | Random |
|--------|-------------|-------------------|--------|
| SR | **36%** (avg) | 75% | 3.84% |
| SPL | **0.25** | 0.55 | 0.023 |
| Episodes | 5,708 | 9,105 | 834 |

### Per-Goal Breakdown

| Goal | SR | Reward | Instances | Mean Geo Dist | Geo/Euc Ratio |
|------|-----|--------|-----------|---------------|---------------|
| plant | **66%** | 6.41 | 1 | 4.13m | 1.18 |
| bed | **59%** | 4.88 | 2 | 2.62m | 1.11 |
| chair | **46%** | 2.91 | 6 | 2.95m | 1.15 |
| sofa | **38%** | 2.53 | 1 | 4.28m | 1.20 |
| toilet | **11%** | -2.19 | 1 | 3.79m | 1.29 |
| tv_monitor | **3%** | -3.69 | 2 | 4.23m | 1.77 |

### Spatial Analysis of Goal Difficulty

The episode distribution is **uniform** (8,333 per goal) — SR differences are NOT explained by episode frequency. Key spatial factors:

| Category | % Episodes <2m | % Episodes <3m | Geo/Euc Ratio |
|----------|---------------|---------------|---------------|
| plant | 1.1% | 12.9% | 1.18 (direct) |
| bed | 3.9% | 99.3% | 1.11 (very direct) |
| chair | 22.4% | 52.6% | 1.15 (direct) |
| sofa | 11.6% | 35.8% | 1.20 (moderate) |
| toilet | 6.8% | 16.7% | 1.29 (indirect) |
| tv_monitor | 5.0% | 8.2% | **1.77** (very indirect) |

**Geo/Euc ratio** = geodesic distance / euclidean distance. Measures path indirectness:
- 1.0 = straight-line path (no obstacles)
- 1.77 (tv_monitor) = path is 77% longer than straight-line, requiring navigation around walls/through doorways

### World Model Losses

| Loss | Train | Val |
|------|-------|-----|
| Dynamics (KL) | 6.71 | 35.0 |
| Reward | 0.25 | 0.17 |
| Value | 1.32 | 2.77 |
| Total | 12.40 | 60.0 |

## Findings

### 1. Clear goal difficulty hierarchy: plant (66%) > bed (59%) > chair (46%) > sofa (38%) > toilet (11%) > tv_monitor (3%)

Without goal conditioning, the agent develops a general navigation policy that succeeds more on some goals than others. The hierarchy correlates with **navigation complexity** (Geo/Euc ratio), NOT with object instance count or episode frequency.

### 2. Navigation complexity (Geo/Euc ratio) is the strongest predictor

- **Easy goals** (plant, bed, chair): Geo/Euc < 1.2 — nearly straight-line paths, objects in open/accessible areas
- **Hard goals** (toilet, tv_monitor): Geo/Euc > 1.29 — paths require navigating around obstacles, through doorways, into enclosed rooms
- tv_monitor at 1.77 is an extreme outlier — near-random SR (3%) despite having 2 instances

### 3. Instance count does not predict SR

Plant has the highest SR (66%) with only 1 instance. Chair has 6 instances but only 46% SR. The house layout and object accessibility matter more than how many instances exist.

### 4. Open question: what makes plant so accessible?

Plant has high SR (66%) despite long mean distance (4.13m) and only 1 instance. The low Geo/Euc ratio (1.18) suggests it's in an open area with direct paths. **Semantic floor maps from Habitat are needed to confirm** — see GitHub issue.

## Next

- **Render semantic floor plans** on cluster to visualize object locations and understand the goal difficulty hierarchy (requires GPU / Habitat renderer)
- Consider adding **goal conditioning** — the agent currently can't distinguish which object to find
- Compare L2 chair SR (46%) with L1 chair SR (75%) — the multi-goal setting reduces per-goal performance
