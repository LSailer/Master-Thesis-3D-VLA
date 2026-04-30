---
run_path: output/runs/r2dreamer-curriculum-l1/_blessed/l1-baseline-2.4m
slurm_id: 3923812
wandb_id: krokhgwi
status: blessed
---

# L1 Baseline — 2.4M Steps, 1 House, Chair Only

**Status**: completed
**Date**: 2026-04-15
**Tags**: #baseline #r2dreamer #habitat #l1 #curriculum #no-goal
**Wandb**: [r2d-L1-1house-chair-3923812](https://wandb.ai/sailer-luca-university-ulm/3d-vla-objectnav/runs/krokhgwi)
**SLURM Job ID**: 3923812
**Slides**: [l1-baseline-2.4m.html](../../l1-baseline-2.4m.html)

## Setup

R2-Dreamer trained for 2.4M environment steps on the L1 curriculum: single house (`fK2vEV32Lag`), chair goal only, no goal conditioning. 64x64 RGB observations, geodesic-delta reward (no step penalty — run predates the configurable rewards change), 500 max steps per episode.

**Hypothesis**: With a single house and one goal category, the world model should learn the scene dynamics and the agent should navigate to chairs significantly above the random baseline (3.84% SR).

## Changes

Compared to the all-scenes baseline (2.36% SR, no learning): restricted to L1 curriculum (1 house, 1 goal) which provides 7,499 training episodes in a single environment — 227x more data per scene.

## Results

| Metric | R2-Dreamer L1 | Random Baseline | Improvement |
|--------|--------------|-----------------|-------------|
| Success Rate | **67%** (rolling) | 3.84% | 17.4x |
| SPL | **0.49** | 0.023 | 21.3x |
| Mean Reward | **9.05** (last 100) | -4.40 | +13.45 |
| Mean Steps | **~250** (last 100) | 489 | 2x faster |
| Episodes | 7,806 total | 834 eval | — |

### Learning Curve

SR rises rapidly in the first 200K steps, reaching ~60% by 500K steps. It fluctuates between 55-75% for the remainder of training, without clear further improvement.

![Success Rate vs Random](../../output/figures/l1-baseline-sr.png)
![SPL vs Random](../../output/figures/l1-baseline-spl.png)
![Episode Metrics](../../output/figures/l1-baseline-episodes.png)

### World Model Losses

![World Model Losses](../../output/figures/l1-baseline-wm-losses.png)
![Policy and KL](../../output/figures/l1-baseline-policy-kl.png)

## Findings

### 1. The agent learns — SR 67% vs 3.84% random

The L1 curriculum works. Restricting to a single house with one goal category allows the world model to learn useful scene dynamics and the policy to discover navigation behavior. This is a 17x improvement over the random baseline and confirms the curriculum hypothesis from the all-scenes failure.

### 2. Why does val/loss/dyn increase?

The validation dynamics loss (KL divergence) rises from 17 to 42 throughout training. This is **overfitting to the training trajectory distribution**. The training episodes cycle through 7,499 starting positions, and the world model memorizes the dynamics from those specific trajectories. Validation episodes have different starting positions, so the model's dynamics predictions are increasingly wrong for unseen states. The train dyn loss also rises (2.9 → 6.0) but much less — the model is being pulled between memorizing training trajectories and maintaining general dynamics understanding.

### 3. Why is loss/policy negative?

This matches the original R2-Dreamer authors' code exactly. The policy loss formula is `-(logpi * advantage + entropy_coeff * entropy)`. The negative sign converts return maximization into a minimization objective for gradient descent. When the agent assigns high probability to high-advantage actions, `logpi * adv` is negative (log-probs are always negative), so `-(negative)` becomes positive — but the entropy term and normalization can push the total below zero.

**Key details:**
- The original R2-Dreamer (PyTorch) uses the same formula at `dreamer.py:470`
- Both use a single optimizer for all parameters — the negative policy loss does reduce the logged `total_loss`
- However, `.detach()` / `stop_gradient` on `weight`, `advantage`, and `imag_feat` ensures gradients only reach the **actor parameters**, not the world model
- The impact on `total_loss` is cosmetic: -0.03 out of 10.5 total (0.34%) — functionally harmless
- The official DreamerV3 (separate optimizers) avoids this confusion, but R2-Dreamer intentionally uses a single-optimizer architecture

### 4. Why does train dyn loss (KL divergence) increase?

The KL divergence between the RSSM prior and posterior rises from 2.9 → 6.0 nats. This means the **posterior is learning richer representations** than the prior can predict. As the encoder gets better at extracting information from observations (posterior), the prior (which only sees the previous latent state and action) falls behind. This is a known DreamerV3 behavior — the `kl_free` parameter (set to 1.0 nat) prevents the KL from being driven to zero, and the increasing gap reflects the posterior gaining information advantage. The fact that the agent still achieves 67% SR despite rising KL suggests the prior is "good enough" for policy optimization, even if imperfect.

### 5. Reward prediction converges fast

Train reward loss drops from 5.4 → 0.03 within 100K steps and stays near zero. The world model quickly learns to predict the geodesic-delta reward signal in this single-house setting. Validation reward loss is noisier (0.1-0.5) but stays bounded.

### 6. Episode length decreases — agent gets efficient

Mean episode length drops from 500 (max, timeout) to ~250 steps. Successful episodes are much shorter, indicating the agent learns purposeful navigation rather than random wandering. The remaining ~250 average includes both fast successes and slow failures.

## Next

- **Run the random baseline evaluation on the same episodes** for direct comparison (already done: 3.84% SR)
- **Add goal conditioning** to see if SR can push beyond 67%
- **Progress to L2 curriculum** (1 house, 6 goals) — test whether the world model transfers across goal categories
- **Investigate overfitting**: the train-val dyn loss gap (6 vs 42) suggests the model could benefit from regularization or data augmentation
- Consider reducing `train_ratio` (currently 512) to reduce overfitting pressure
