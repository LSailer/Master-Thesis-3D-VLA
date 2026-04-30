# REINFORCE — Williams (1992)

**Source:** Williams, R.J., "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning," *Machine Learning*, 8, 229–256 (1992).

## Core Idea

REINFORCE is a class of policy gradient algorithms for stochastic neural networks. The key insight: weight updates that follow the **characteristic eligibility** — $\partial \ln g_i / \partial w_{ij}$ (the gradient of the log-probability of the unit's output w.r.t. its weights) — are guaranteed to follow the gradient of expected reinforcement, **without ever computing or storing that gradient explicitly**.

The name is an acronym: **RE**ward **I**ncrement = **N**onnegative **F**actor × **O**ffset **R**einfor**ce**ment × **C**haracteristic **E**ligibility.

## The REINFORCE Update Rule

Each weight $w_{ij}$ is updated by:

$$\Delta w_{ij} = \alpha_{ij}(r - b_{ij})\, e_{ij}$$

where:
- $\alpha_{ij} \geq 0$ — learning rate factor
- $r$ — scalar reinforcement signal (reward)
- $b_{ij}$ — **reinforcement baseline** (conditionally independent of $y_i$ given $\mathbf{w}^i$ and $\mathbf{x}^i$)
- $e_{ij} = \partial \ln g_i / \partial w_{ij}$ — **characteristic eligibility** (score function of the unit's output distribution)

## Theorem 1 (Unbiased Gradient)

For any REINFORCE algorithm, the inner product of $E\{\Delta \mathbf{W} \mid \mathbf{W}\}$ and $\nabla_\mathbf{W} E\{r \mid \mathbf{W}\}$ is nonnegative. If $\alpha_{ij} = \alpha$ for all $i, j$:

$$E\{\Delta \mathbf{W} \mid \mathbf{W}\} = \alpha\, \nabla_\mathbf{W} E\{r \mid \mathbf{W}\}$$

**Meaning:** the average weight update is exactly proportional to the gradient of expected reward. Each individual update $(r - b_{ij})\, \partial \ln g_i / \partial w_{ij}$ is an **unbiased estimate** of $\partial E\{r \mid \mathbf{W}\} / \partial w_{ij}$.

## Concrete Instantiations

### Bernoulli-logistic units (discrete actions)
For a unit with $y_i \in \{0, 1\}$, $p_i = \sigma(s_i)$:

$$\frac{\partial \ln g_i}{\partial w_{ij}} = (y_i - p_i)\, x_j \tag{7}$$

With baseline $b = 0$, the update becomes: $\Delta w_{ij} = \alpha\, r\, (y_i - p_i)\, x_j$ (Eq. 8). This recovers the associative reward-inaction ($A_{R\text{-}I}$) algorithm of Barto & Anderson (1985).

### Gaussian units (continuous actions)
For a unit outputting $y \sim \mathcal{N}(\mu, \sigma^2)$:

$$\frac{\partial \ln g}{\partial \mu} = \frac{y - \mu}{\sigma^2} \tag{12}$$

$$\frac{\partial \ln g}{\partial \sigma} = \frac{(y - \mu)^2 - \sigma^2}{\sigma^3}$$

Key property: $\sigma$ controls exploration — it naturally narrows around good solutions and widens when stuck on flat regions.

## Reinforcement Baseline

The baseline $b$ does not bias the gradient (Theorem 1 holds for any valid $b$) but **reduces variance**. Common choices:
- $b = 0$ — simplest, high variance
- **Reinforcement comparison** — adaptive estimate $\bar{r}$ via exponential averaging: $\bar{r}(t) = \gamma\, r(t-1) + (1-\gamma)\, \bar{r}(t-1)$ (Eq. 10)
- **Optimal baseline** — minimizes variance of weight updates (Dayan 1990): related to expected reinforcement but not exactly equal to it

## Episodic REINFORCE (Section 5)

For tasks with delayed reward delivered at episode end, the network is "unfolded in time" over $k$ steps. The update sums eligibilities across the episode:

$$\Delta w_{ij} = \alpha_{ij}(r - b_{ij}) \sum_{t=1}^{k} e_{ij}(t) \tag{11}$$

**Theorem 2** extends Theorem 1 to episodic REINFORCE — the same unbiased gradient property holds.

## Integration with Backpropagation (Section 7)

REINFORCE composes with backprop: for a multi-layer network with stochastic units, the characteristic eligibility of an internal weight can be computed via the chain rule through deterministic layers. This means REINFORCE naturally extends to deep networks — the same backprop machinery computes $\partial \ln g / \partial w$ for hidden-layer weights.

## Limitations (Section 8–9)

- **No convergence guarantee** — expected updates follow the gradient, but convergence to global optima is not proven (local optima, like all gradient methods)
- **High variance** — individual updates are noisy; baselines help but don't eliminate this
- **Slow with sparse/delayed rewards** — episodic REINFORCE spreads credit uniformly across time steps; temporal-difference methods (actor-critic) are more sample-efficient for sequential tasks
- **Not model-based** — REINFORCE does not build or use a model of the environment

## Connection to DreamerV3

DreamerV3 (Hafner et al. 2024) uses REINFORCE as its actor loss (Eq. 6 in that paper), adapted for world-model imagination:
- The **characteristic eligibility** becomes $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$
- The **reinforcement signal** is the lambda-return $R_t^\lambda$ with critic baseline $v_\psi(s_t)$
- The **baseline** is the critic value function (a learned, state-dependent version of Williams' $\bar{r}$)
- Returns are normalized by percentile range (replacing simple averaging schemes like Eq. 10)

See [DreamerV3 Policy Loss](dreamerv3-policy-loss.md) for the full formulation.
