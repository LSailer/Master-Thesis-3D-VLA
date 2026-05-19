---
run_path: output/runs/baselines/_blessed/random-baseline-l1
slurm_id: 3924640
wandb_id: unknown
status: blessed
---

# Random Baseline — L1 Curriculum

**Status**: completed
**Date**: 2026-04-13
**Tags**: #baseline #random #l1 #curriculum #objectnav
**SLURM Job ID**: 3924640
**Slides**: [random-baseline.html](../../random-baseline.html)

## Setup

Establishes the random-chance performance floor for the L1 curriculum (1 house: `fK2vEV32Lag`, chair only). A uniform-random agent selects from 4 actions (STOP, FORWARD, LEFT, RIGHT) with equal probability on all 834 eval episodes, 500 max steps per episode. Reward uses geodesic delta + step penalty (-0.01/step) + success bonus (10.0).

**Hypothesis**: Random performance should be low (SR < 5%) and serve as the minimum bar that any trained agent must exceed.

## Changes

New standalone module `src/baselines/random_agent.py` — no model, pure random actions. Also added configurable reward parameters (`step_penalty`, `success_bonus`) to `DreamerConfig` and `R2DreamerConfig`.

## Results

| Metric | Value |
|--------|-------|
| Success Rate | 3.84% (32/834) |
| SPL | 0.023 |
| Mean Reward | -4.40 ± 2.95 |
| Mean Steps | 489 ± 62 |
| Action Distribution | ~25% each (uniform) |

### Plots

![Action Distribution](../../output/figures/random-l1-action-dist.png)
![Reward Distribution](../../output/figures/random-l1-reward-dist.png)
![Episode Length](../../output/figures/random-l1-episode-length.png)
![Success Scatter](../../output/figures/random-l1-success-scatter.png)

## Findings

1. **3.84% SR is the random floor** — 32 out of 834 episodes succeeded by chance. This is higher than the 2.36% from the all-scenes baseline because L1 uses a single house with many chair instances, increasing the probability of random proximity.

2. **Mean reward decomposes cleanly**: step penalty dominates at -0.01 × 489 = -4.89, partially offset by small positive geodesic deltas from random forward movements (~+0.49 on average). Successful episodes show rewards of +7 to +12 (success bonus overcomes the step penalty).

3. **Almost all episodes time out** — the episode length distribution is a spike at 500 steps. Only successful episodes (32) terminate early, some as early as 30-50 steps (lucky spawns near a chair).

4. **Bimodal reward distribution** — a large cluster at -5 (failures with step penalty) and a small cluster at +7 to +12 (successes with bonus). No middle ground, confirming that random actions don't produce partial progress.

## Next

- Train R2-Dreamer on L1 curriculum and compare against this baseline
- The trained agent should show: SR >> 3.84%, SPL >> 0.023, shorter episodes, non-uniform action distribution
- If the trained agent's SR is not significantly above 3.84%, the world model is not learning useful navigation behavior
