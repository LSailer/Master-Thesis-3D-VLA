# Baseline 2.4M — All Scenes, No Goal Conditioning

**Status**: completed
**Date**: 2026-04-11
**Tags**: #baseline #r2dreamer #habitat #no-goal #all-scenes
**Wandb**: [r2dreamer-baseline-2.4M-3907457](https://wandb.ai/sailer-luca-university-ulm/3d-vla-objectnav/runs/qwdqowxq)
**SLURM Job ID**: 3907457
**Slides**: [baseline-2.4m-all-scenes.html](../../baseline-2.4m-all-scenes.html)

## Setup

R2-Dreamer trained for 2.4M environment steps on HM3D ObjectNav with **all 145 training scenes** and **all 6 goal categories**, using geodesic-delta reward and no goal conditioning. The agent receives only 64×64 RGB observations — no information about which object to find.

**Hypothesis**: Without goal conditioning and with massive scene diversity, the agent will fail to learn meaningful navigation behavior. This establishes the lower bound for our curriculum experiments.

## Changes

First full-scale Habitat training run. No prior baselines to compare against within this project.

## Results

- **4,871 episodes** over ~18.6 hours on a single H100
- **115 successes (2.36% SR)** — flat across training, no improvement
- **0 SPL** (wandb summary), mean episode length 493/500 steps (hitting max)
- **19 NaN rewards** (minor numerical issue)

| Quartile | Steps | Episodes | Successes | SR |
|----------|-------|----------|-----------|-----|
| Q1 | 0–600K | 1,217 | 32 | 2.6% |
| Q2 | 600K–1.2M | 1,217 | 35 | 2.9% |
| Q3 | 1.2M–1.8M | 1,217 | 27 | 2.2% |
| Q4 | 1.8M–2.4M | 1,220 | 21 | 1.7% |

### World Model Losses

- **KL divergence increases** from 2.3 → 10.2 over training — world model fails to converge on dynamics across 145 diverse scenes
- **Reward loss converges** quickly (5.4 → 0.04) — trivial prediction since reward is nearly constant
- **Policy loss flat** at -0.04 — no policy gradient signal

### Action Distribution

Nearly uniform: forward 27.6%, left 24.2%, right 27.2%, stop 21.0% — effectively random.

![Episode Metrics](../../output/figures/baseline_episode_metrics.png)
![World Model Losses](../../output/figures/baseline_wm_losses.png)
![Policy Diagnostics](../../output/figures/baseline_policy_diagnostics.png)

## Findings

**The model does not learn.** Three compounding factors explain why:

1. **No goal conditioning**: The agent has no information about which object to navigate to. With 6 goal categories across 145 houses, the geodesic-delta reward is effectively noise from the agent's perspective — moving toward a chair in one episode is the wrong direction for a toilet in the next.

2. **Scene diversity overwhelms the world model**: The KL divergence *increases* throughout training. The RSSM cannot model the dynamics of 145 visually distinct environments simultaneously at this model scale. The world model never converges, so imagination-based policy optimization has no useful signal.

3. **Sparse useful reward signal**: The 10.0 success bonus occurs 2.36% of the time (by chance), while per-step geodesic deltas are tiny (~0.001). The reward predictor learns to predict "approximately zero" and stops there.

The 2.36% success rate is consistent with random exploration occasionally stumbling onto nearby goals. The slight decrease in later quartiles (2.6% → 1.7%) may indicate the policy learning counterproductive movement patterns.

> **Relevance to research question**: This confirms that naively training a world model on diverse environments without goal conditioning produces no useful behavior. The curriculum approach (restricting scenes and goals) is essential for learning.

## Next

1. **Fix L1 curriculum bug** (episode ID mismatch) and rerun L1 (1 house, chair only)
2. L1 should isolate whether the world model can learn navigation in a constrained setting
3. If L1 succeeds → progress through L2–L4 curriculum levels
4. Consider adding goal conditioning (e.g., goal image or category embedding) for multi-goal settings
