# DreamerV3 Policy Loss

**Source:** Hafner et al., "Mastering Diverse Domains through World Models" (2024)

## Overview

DreamerV3 trains actor and critic purely from imagined trajectories in the learned world model — no real environment interaction during actor-critic updates. The actor and critic operate on model states $s_t \doteq \{h_t, z_t\}$ (deterministic recurrent state + stochastic representation).

## Actor (Policy) Loss

The actor maximizes return while exploring via an entropy regularizer. DreamerV3 uses the **REINFORCE** estimator for both discrete and continuous actions, with a **return-normalized advantage** baseline:

$$\mathcal{L}(\theta) = -\sum_{t=1}^{T-1} \text{sg}\!\Big(\frac{R_t^\lambda - v_\psi(s_t)}{\max(1,\, S)}\Big) \cdot \log \pi_\theta(a_t \mid s_t) \;+\; \eta\, H[\pi_\theta(a_t \mid s_t)] \tag{6}$$

Key components:

- **Advantage:** $R_t^\lambda - v_\psi(s_t)$ — lambda-return minus critic baseline, stop-gradiented (`sg`)
- **Return normalization S:** percentile range smoothed with EMA:
  $$S = \text{EMA}\!\big(\text{Per}(R_t^\lambda, 95) - \text{Per}(R_t^\lambda, 5),\; 0.99\big) \tag{7}$$
  Normalizes returns to ≈[0, 1] regardless of reward scale. Uses 5th–95th percentile (not min/max) to be robust to outliers.
- **Entropy regularizer:** $\eta\, H[\pi_\theta]$ with fixed scale $\eta = 3 \times 10^{-4}$ across all domains. Encourages exploration.
- **Prediction horizon:** $T = 16$ imagined steps.

### Why this design

1. **REINFORCE over reparameterization:** works for both discrete and continuous action spaces with the same code.
2. **Return normalization over reward normalization:** normalizing rewards or returns by standard deviation fails under sparse rewards (std ≈ 0 → amplified noise). Percentile-based normalization targets a fixed entropy of exploration regardless of reward magnitude.
3. **Denominator clamp** $\max(1, S)$: only scales down large return magnitudes, leaves small returns below $L = 1$ untouched — avoids amplifying noise from function approximation under sparse rewards.

## Critic Loss

The critic learns to predict the **distribution** of lambda-returns $R_t^\lambda$ (not just the mean), using maximum likelihood:

$$\mathcal{L}(\psi) = -\sum_{t=1}^{T} \ln p_\psi(R_t^\lambda \mid s_t) \tag{5}$$

where the lambda-return is:

$$R_t^\lambda = r_t + \gamma c_t \big((1-\lambda)\, v_t + \lambda\, R_{t+1}^\lambda\big), \qquad R_T^\lambda = v_T \tag{5}$$

- **Distributional critic:** parameterized as categorical distribution over exponentially spaced bins (symexp twohot), not a Gaussian — handles multi-modal return distributions and varying scales across domains.
- **Discount factor:** $\gamma = 0.997$
- **Slow regularization:** critic regresses toward an EMA of its own parameters to stabilize learning.

## Symlog & Twohot (Robust Predictions)

DreamerV3 uses **symlog** transformations throughout to handle quantities of unknown magnitude:

$$\text{symlog}(x) = \text{sign}(x)\ln(|x| + 1), \qquad \text{symexp}(x) = \text{sign}(x)(\exp(|x|) - 1)$$

For the critic and reward predictor, predictions use **symexp twohot loss** (Eq. 10–11):
- Network outputs logits for a softmax over exponentially spaced bins $b_i \in B$, $B = \text{symexp}([-20 \ldots +20])$
- Prediction: $\hat{y} = \text{softmax}(f(x))^T B$
- Trained on twohot-encoded targets with categorical cross-entropy: $\mathcal{L}(\theta) = -\text{twohot}(y)^T \log \text{softmax}(f(x, \theta))$
- Decouples gradient magnitude from target magnitude — large targets don't cause large gradients.
