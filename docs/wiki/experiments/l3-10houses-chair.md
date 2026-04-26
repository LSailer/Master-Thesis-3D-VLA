---
run_path: output/runs/r2dreamer-curriculum-l3/_blessed/l3-10houses-chair
slurm_id: 3957714
wandb_id: rsopsua1
status: blessed
---

# L3 — 10 Houses, Chair Only, 2.4M Steps

**Status**: completed
**Slides**: [curriculum-scaling.html](../../curriculum-scaling.html)
**Date**: 2026-04-16
**Tags**: #r2dreamer #habitat #l3 #curriculum #generalization #buffer-fix
**Wandb**: [r2d-L3-buffix-3957714](https://wandb.ai/sailer-luca-university-ulm/3d-vla-objectnav/runs/rsopsua1)
**SLURM Job ID**: 3957714

## Setup

R2-Dreamer trained for 2.4M environment steps on the L3 curriculum: **10 houses** (4 easy, 4 medium, 2 hard), chair only, no goal conditioning. 64x64 RGB observations, geodesic-delta reward with step penalty (-0.01/step) and success bonus (10.0), 1000 max steps per episode.

Houses: fK2vEV32Lag (easy), W9YAR9qcuvN (easy), wPLokgvCnuk (easy), ACZZiU6BXLz (easy), XfUxBGTFQQb (medium), 9h5JJxM6E5S (medium), qz3829g1Lzf (medium), oPj9qMxrDEa (medium), u5atqC7vRCY (hard), j2EJhFEQGCL (hard).

**Hypothesis**: With 10 houses but only 1 goal category (chair), the agent should learn above random but below L1 (single house). This tests whether the world model can generalize dynamics across diverse environments.

## Changes

Compared to L1 rerun (75% SR, 1 house, chair):
- **10 houses** instead of 1 — same goal (chair), same architecture, same hyperparameters
- Data is spread across 10 environments instead of concentrated in 1

## Configuration

Same as L1 rerun except:
| Parameter | Value |
|-----------|-------|
| Curriculum | level3_10houses_1goal |
| Scenes | 10 (4 easy, 4 medium, 2 hard) |
| Goals | chair only |

## Results

| Metric | L3 (10 houses) | L1 Rerun (1 house) | Random |
|--------|----------------|---------------------|--------|
| SR | **32%** | 75% | 3.84% |
| SPL | **0.21** | 0.55 | 0.023 |
| Episodes | 5,509 | 9,105 | 834 |
| Mean Reward | 0.82 (rolling) | 7.40 | -4.40 |

### World Model Losses

| Loss | Train | Val |
|------|-------|-----|
| Dynamics (KL) | 6.89 | 27.49 |
| Reward | 0.25 | 0.21 |
| Value | 1.61 | 3.55 |
| Total | 12.72 | 42.09 |

### Action Distribution

forward 27.6%, left 27.6%, right 22.1%, stop 22.7% — more balanced than L1, less STOP-heavy.

## Findings

### 1. Generalization costs ~43pp SR: 75% (1 house) → 32% (10 houses)

The agent still learns well above random (32% vs 3.84% = 8.3x), but spreading data across 10 visually distinct houses significantly reduces performance. With 2.4M total steps, each house sees ~240K steps — compared to 2.4M concentrated in one house for L1.

### 2. Val dyn loss is actually lower than L1 (27.5 vs 40.2)

Despite lower SR, the validation dynamics loss is lower. This makes sense: with 10 diverse houses, the world model is forced to learn more general dynamics rather than memorizing a single environment. Less overfitting, but also less specialized performance.

### 3. Fewer episodes (5,509 vs 9,105)

Fewer episodes in the same step budget suggests the agent times out more often in unfamiliar houses, consistent with the lower SR.

## Next

- Investigate **per-house SR** to understand which houses are easier/harder (need cluster access)
- Compare L1 + L3 to understand the **data efficiency vs generalization tradeoff**
- Progress to L4 (10 houses, 6 goals) as the full curriculum test
