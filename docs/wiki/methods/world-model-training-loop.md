# World Model Training Loop — How 64-Step Windows Produce 200-Step Episodes

## Overview

R2-Dreamer (and DreamerV3) separates **acting** from **training**. The world model learns from short replay windows, but the agent executes full-length episodes. Understanding this separation is key to understanding why the architecture works.

## The Three Phases

### 1. Acting (Full Episodes)

The agent interacts with the environment for **up to 500 steps** per episode. At each step it picks an action using its learned policy. Successful episodes take ~200 steps on average (L1 baseline). All transitions `(obs, action, reward, done)` are stored sequentially in a flat ring-buffer replay.

### 2. World Model Training (64-Step Windows)

The replay buffer samples **random 64-step windows** — contiguous subsequences from the flat buffer. These windows can span episode boundaries; the `is_first` flag marks where new episodes begin, and the RSSM resets its hidden state (stoch, deter → zeros) at those points.

The world model learns **local dynamics** from these windows:
- Encoder: obs → embedding
- RSSM posterior: embedding + prev_state + action → next_state (learns transition dynamics)
- Reward head: state → predicted reward
- Continuation head: state → predicted terminal

64 steps is sufficient because the model learns step-by-step transitions ("FORWARD moves you ahead", "LEFT rotates the view"), not full episode strategies.

### 3. Policy Training (15-Step Imagination)

The actor and critic train on **imagined trajectories** — not on real replay data directly:
1. Sample states from the 64-step replay window (the "dreaming" starting points)
2. Roll out 15 steps forward using the world model's prior (no encoder, no real observations)
3. The actor learns which actions maximize imagined reward over 15 steps
4. The critic learns expected returns from each imagined state

### How Full Episodes Emerge

At test time, the agent applies its 15-step planning horizon **at every single step** for 200+ steps. Each decision is locally optimal over a 15-step lookahead, and composing hundreds of these short-horizon decisions produces coherent long-horizon behavior (navigating to a goal).

**Analogy:** You don't practice driving a full 100km route to learn to drive. You learn local skills (steering, braking, turning) from short practice segments and compose them at execution time.

## Why Fixed Windows (Not Full Episodes)

| Aspect | 64-step windows | Full episodes |
|--------|-----------------|---------------|
| **JAX JIT** | Fixed shape → efficient compilation | Variable length (50–500) → needs padding |
| **Data diversity** | ~186 windows per 250-step episode | 1 sample per episode |
| **Batch efficiency** | No wasted padding | Short episodes waste padding in long batches |
| **Dynamics learning** | 64 steps captures all local transitions | No additional benefit for transition learning |

## Key Numbers (L1 Baseline)

| Parameter | Value |
|-----------|-------|
| `seq_len` | 64 (training window) |
| `imagination_horizon` | 15 (policy planning) |
| `max_episode_steps` | 500 (acting timeout) |
| Mean episode length | ~200 (successful L1 episodes) |
| `batch_size` | 16 (parallel sequences per training step) |
| `train_ratio` | 512 (training frames per env step) |
