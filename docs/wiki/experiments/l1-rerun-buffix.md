# L1 Rerun — Buffer Fix + Step Penalty, 2.4M Steps

**Status**: analysis-in-progress
**Date**: 2026-04-16
**Tags**: #baseline #r2dreamer #habitat #l1 #curriculum #buffer-fix #rerun
**Wandb**: [r2d-L1-buffix-3957651](https://wandb.ai/sailer-luca-university-ulm/3d-vla-objectnav/runs/y5a0upzd)
**SLURM Job ID**: 3957651

## Setup

R2-Dreamer trained for 2.4M environment steps on the L1 curriculum: single house (`fK2vEV32Lag`), chair only, no goal conditioning. 64x64 RGB observations, geodesic-delta reward with step penalty (-0.01/step) and success bonus (10.0), **1000 max steps** per episode.

**Hypothesis**: With the buffer fix (RFC #68) and step penalty, the L1 result should reproduce or improve upon the original L1 baseline (67% SR, 0.49 SPL).

## Changes

Compared to the original L1 baseline (run 3923812, wandb krokhgwi):
- **Buffer fix**: unified ReplayBuffer from RFC #68 — fixes replay sampling issues
- **Step penalty**: -0.01/step incentivizes shorter paths
- **Max episode steps**: 1000 (was 500) — more exploration room
- **Seed**: 3957651 (was 3923812)

## Configuration

| Parameter | Value |
|-----------|-------|
| total_steps | 2,400,000 |
| batch_size | 16 |
| seq_len | 64 |
| train_ratio | 512 |
| imagination_horizon | 15 |
| lr | 4e-05 |
| deter_size | 2048 |
| stoch_discrete | 16 × 32 |
| step_penalty | -0.01 |
| success_bonus | 10 |
| max_episode_steps | 1000 |
| buffer_capacity | 1,000,000 |

## Results

| Metric | L1 Rerun | Original L1 | Random Baseline | vs Original |
|--------|----------|-------------|-----------------|-------------|
| Success Rate | **75%** | 67% | 3.84% | +8pp |
| SPL | **0.55** | 0.49 | 0.023 | +0.06 |
| Mean Reward | **7.40** (rolling) | 9.05 | -4.40 | — |
| Episodes | 9,105 | 7,806 | 834 | +17% |

### Interpretation

- **SR improvement (67% → 75%)**: primarily attributed to the buffer fix improving replay diversity
- **SPL improvement (0.49 → 0.55)**: step penalty incentivizes shorter paths, reducing frivolous STOP actions
- **More episodes (7,806 → 9,105)**: 1000 max steps allows longer exploration, but the agent also finishes faster on average due to the step penalty

### World Model Losses

| Loss | Train | Val |
|------|-------|-----|
| Dynamics (KL) | 5.92 | 40.19 |
| Reward | 0.27 | 0.21 |
| Value | 1.46 | 2.30 |
| Total | 11.43 | 68.0 |

Overfitting pattern persists: val dyn loss (40.2) >> train dyn loss (5.9), consistent with the original L1 finding.

## Findings

### 1. Buffer fix + step penalty improve L1 from 67% to 75% SR

The buffer fix is the primary driver of SR improvement — better replay sampling means the world model sees a more representative distribution of experiences. The step penalty contributes mainly to SPL improvement by discouraging unnecessary steps.

### 2. Overfitting persists

The train-val dynamics loss gap remains large (5.9 vs 40.2), similar to the original L1. This is expected — the single-house setting provides extensive data but the world model still overfits to training trajectory distributions.

## Next

- Use this as the updated L1 baseline for comparison with L2 and L3
- The buffer fix and step penalty are now standard for all subsequent runs
