# Spec: JAX DreamerV3 Baseline for HM3D ObjectNav

## Context

Issue #6 requests a vanilla DreamerV3 baseline trained on HM3D ObjectNav. The March 3 meeting pivoted the thesis from diffusion policies to a **Dreamer-style world model**. This spec defines a simple JAX/Equinox implementation modeled after [NaturalDreamer](https://github.com/InexperiencedMe/NaturalDreamer)'s clean structure, but rewritten in JAX to align with the project's existing JAX/Equinox/Optax stack (`pixi.toml` lines 45-47).

**Goal**: Minimal, readable DreamerV3 that trains on HM3D ObjectNav observations (RGB 256x256, depth 256x256, discrete 4-action space) and logs to W&B. This is the 2D-only baseline; 3D VGGT augmentation comes later.

## Architecture (DreamerV3 RSSM)

Follows the same component structure as NaturalDreamer, translated to JAX/Equinox:

```
Observation (RGB+depth) → Encoder (CNN) → Posterior (MLP) → z_t (categorical latent)
                                              ↑
                          Recurrent (GRU) → h_t (deterministic state)
                              ↑
                     Prior (MLP) → ẑ_t (predicted latent)
                              ↑
                        h_{t-1}, a_{t-1}

Decoder: (h_t, z_t) → reconstructed obs
Reward:  (h_t, z_t) → predicted reward
Continue:(h_t, z_t) → episode continuation probability
Actor:   (h_t, z_t) → action logits (discrete, 4 classes)
Critic:  (h_t, z_t) → value estimate
```

## File Structure (new files)

```
src/dreamer/
├── __init__.py
├── networks.py        # All network modules (Equinox)
├── agent.py           # Dreamer agent: world model + actor-critic training
├── replay_buffer.py   # Simple numpy replay buffer
├── envs.py            # Habitat ObjectNav wrapper → Gymnasium-like interface
├── config.py          # Dataclass config with defaults
└── train.py           # Main training loop entry point

configs/
└── dreamer_objectnav.yaml   # HM3D ObjectNav hyperparameters

tests/
└── test_dreamer.py    # Shape tests + toy overfit test
```

## Implementation Details

### 1. `src/dreamer/config.py` — Configuration dataclass

```python
@dataclasses.dataclass
class DreamerConfig:
    # RSSM
    recurrent_size: int = 512
    latent_length: int = 32
    latent_classes: int = 32
    hidden_size: int = 512
    num_layers: int = 2
    # CNN encoder/decoder
    cnn_depth: int = 48
    kernel_size: int = 4
    stride: int = 2
    # Training
    batch_size: int = 16
    sequence_length: int = 50
    learning_rate: float = 1e-4
    max_grad_norm: float = 100.0
    # Imagination
    imagination_horizon: int = 15
    gamma: float = 0.997
    lambda_: float = 0.95
    # Losses
    kl_free_nats: float = 1.0
    kl_balance: float = 0.8
    # Replay
    buffer_capacity: int = 1_000_000
    # Environment
    action_size: int = 4           # discrete ObjectNav actions
    obs_shape: tuple = (4, 256, 256)  # RGB(3) + depth(1), channels-first
    # Logging
    wandb_project: str = "3d-vla-objectnav"
    log_interval: int = 100
```

### 2. `src/dreamer/networks.py` — Equinox modules

All modules as `eqx.Module` classes. Key components:

| Module | Input | Output | Notes |
|--------|-------|--------|-------|
| `ConvEncoder` | `[C,H,W]` image | `[embed_dim]` vector | 4-layer CNN, SiLU activation |
| `ConvDecoder` | `[state_dim]` | `[C,H,W]` reconstruction | Transposed CNN |
| `RecurrentModel` | `(h, z, a)` | `h'` | GRU cell |
| `PriorNet` | `h` | `(z_sample, logits)` | Categorical + uniform mix |
| `PosteriorNet` | `(h, embed)` | `(z_sample, logits)` | Categorical + uniform mix |
| `RewardModel` | `(h, z)` | `Normal dist` | MLP → mean, logstd |
| `ContinueModel` | `(h, z)` | `Bernoulli dist` | MLP → logit |
| `DiscreteActor` | `(h, z)` | `Categorical dist` | MLP → 4-class logits |
| `Critic` | `(h, z)` | `Normal dist` | MLP → mean, logstd |

Key JAX/Equinox patterns:
- Use `eqx.nn.GRUCell` for recurrence
- Use `jax.nn.silu` activation throughout
- Categorical latent: `jax.random.categorical` + straight-through gradient via stop_gradient trick
- All modules are pytrees — compatible with `jax.jit`, `jax.vmap`

### 3. `src/dreamer/agent.py` — Training logic

Two main JIT-compiled functions:

**`train_world_model(params, opt_state, batch, key)`**:
1. Encode observations → embeddings
2. Unroll RSSM: for each timestep, compute posterior (from h + embed), prior (from h), GRU update
3. Losses: reconstruction (decoder MSE), reward prediction (NLL), continue prediction (NLL), KL divergence (free nats + balancing)
4. Return updated params, opt_state, metrics

**`train_actor_critic(params, opt_state, initial_state, key)`**:
1. Imagine trajectories: unroll prior + GRU for `imagination_horizon` steps using actor policy
2. Predict rewards and values along imagined trajectory
3. Compute lambda-returns
4. Actor loss: policy gradient with entropy regularization
5. Critic loss: value prediction NLL
6. Return updated params, opt_state, metrics

Use `optax.adam` with gradient clipping via `optax.clip_by_global_norm`.

### 4. `src/dreamer/replay_buffer.py` — Simple numpy buffer

Flat numpy arrays (same approach as NaturalDreamer's `buffer.py`):
- Stores: observations, actions, rewards, dones
- `add(obs, action, reward, done)` — circular buffer
- `sample(batch_size, sequence_length)` — sample contiguous sequences
- No JAX dependency — pure numpy, convert to jax arrays at sample time

### 5. `src/dreamer/envs.py` — Habitat wrapper

Wraps Habitat ObjectNav into a simple interface:

```python
class HabitatObjectNavEnv:
    def __init__(self, config_path, split="val_mini"):
        ...
    def reset(self) -> np.ndarray:  # [C, H, W] float32
        ...
    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        ...
    @property
    def observation_shape(self) -> tuple: ...
    @property
    def action_size(self) -> int: ...
```

- Combines RGB (3ch) + depth (1ch) → 4-channel observation, channels-first
- Normalizes RGB to [0,1], depth already float
- Returns standard (obs, reward, done, info) tuple
- Info includes: success, spl, distance_to_goal

### 6. `src/dreamer/train.py` — Entry point

```
python -m src.dreamer.train --config configs/dreamer_objectnav.yaml
```

Training loop:
1. Initialize env, buffer, networks, optimizers
2. Collect `prefill_steps` random actions into buffer
3. Loop:
   a. Act in environment using actor (with exploration noise)
   b. Store transition in buffer
   c. Every N env steps: sample batch, train world model, train actor-critic
   d. Log metrics to W&B every `log_interval` steps
4. Checkpoint saving via `eqx.tree_serialise_leaves`

### 7. `configs/dreamer_objectnav.yaml`

Override defaults for HM3D ObjectNav:
```yaml
obs_shape: [4, 256, 256]
action_size: 4
batch_size: 16
sequence_length: 50
buffer_capacity: 500000
imagination_horizon: 15
total_steps: 1000000
prefill_steps: 5000
train_every: 5
wandb_project: "3d-vla-objectnav"
```

### 8. `tests/test_dreamer.py`

- `test_encoder_output_shape` — CNN encoder produces expected embedding dim
- `test_decoder_output_shape` — decoder reconstructs correct image shape
- `test_rssm_single_step` — one RSSM step produces correct h, z shapes
- `test_rssm_sequence` — unroll over sequence, shapes consistent
- `test_actor_discrete_logits` — actor outputs 4-class logits
- `test_critic_value` — critic outputs scalar value distribution
- `test_replay_buffer_sample` — buffer stores and samples correct shapes
- `test_world_model_loss_decreases` — 10 gradient steps on random data, loss drops

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Framework** | JAX + Equinox + Optax | Already in `pixi.toml`; thesis direction |
| **Discrete actions** | Categorical actor (not continuous) | HM3D ObjectNav uses 4 discrete actions |
| **Observation** | RGB + depth concatenated (4ch) | Simplest fusion; VGGT added later |
| **No symlog** | Skip DreamerV3's symlog transforms | Simplicity; add if needed |
| **No twohot** | Normal distribution for reward/value | NaturalDreamer also skips this; add later |
| **Categorical latent** | 32 × 32 one-hot (as in DreamerV3) | Core to DreamerV3's robustness |

## Files to Modify

| File | Change |
|------|--------|
| `pixi.toml` | Add `pyyaml` to pypi-dependencies (for config loading) |

## Verification

1. **Unit tests**: `pixi run pytest tests/test_dreamer.py -v`
2. **Smoke test**: `pixi run python -m src.dreamer.train --config configs/dreamer_objectnav.yaml --total-steps 100 --prefill-steps 10` (runs 100 steps, no GPU needed for shape validation)
3. **Full training**: Submit via Slurm on BWUniCluster (depends on issues #4, #5)
4. **Metrics**: W&B dashboard shows world model loss decreasing, reward prediction improving, episode returns increasing over training

## Dependencies

- Blocked by: Issue #4 (self-hosted runner), Issue #5 (BWUniCluster env) for actual training
- NOT blocked for: code implementation, unit tests, local smoke tests
