# PR Spec: Diffusion Policy + Flow Matching Action Heads

## Goal

Implement both action head variants (DDPM Diffusion Policy and Conditional Flow Matching) as standalone, swappable modules. Include an exploration notebook that builds intuition from scratch — noise schedules, denoising, velocity fields, then full goal-conditioned action prediction.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Action space** | Discrete first (6 classes), continuous later | Habitat ObjectNav uses `STOP`, `MOVE_FORWARD`, `TURN_LEFT`, `TURN_RIGHT`, `LOOK_UP`, `LOOK_DOWN`. Diffusion/FM only needed if we switch to continuous velocity control |
| **Chunk size** | 4 | Nav episodes ~50-100 steps at ~5Hz. Chunk of 4 = plan 4 ahead, re-plan every 4. Larger chunks (8-16) for high-freq manipulation, overkill for nav |
| **Denoiser arch** | MLP (concat cond + x_t + time_emb, 4 hidden layers). Potential upgrade to Transformer later | Simpler, faster to iterate. Transformer only needed when chunk grows or multi-token goal conditioning |
| **EMA** | Yes | Exponential Moving Average of weights: `ema = 0.999 * ema + 0.001 * weights` after each step. Use EMA weights at inference. Smooths training noise → better samples. ~5 lines of code |
| **Training paradigm** | IL (Imitation Learning) first | Clone expert demonstrations from Habitat's `ShortestPathFollower` oracle. RL fine-tune later (optional, via FMPG) |

### Action Magnitudes

| Action | Index | Physical effect |
|--------|-------|-----------------|
| STOP | 0 | terminate episode |
| MOVE_FORWARD | 1 | +0.25 m forward |
| TURN_LEFT | 2 | +30° yaw |
| TURN_RIGHT | 3 | −30° yaw |
| LOOK_UP | 4 | −30° pitch |
| LOOK_DOWN | 5 | +30° pitch |

## Dataset Prerequisites

Before collecting expert demonstrations:

1. **Install habitat-lab** (added to `pyproject.toml` pip deps)
2. **Download HM3D scene meshes**:
   ```bash
   python -m habitat_sim.utils.datasets_download --data-path data/ --uids hm3d_*
   ```
   Requires Matterport API token in `.env`.
3. **Episode dataset**: `objectnav_hm3d_v2`, path layout:
   ```
   data/datasets/objectnav/hm3d/objectnav_hm3d_v2/{split}/{split}.json.gz
   ```
4. **Observation config**: 640×480 RGB-D, hfov=79°, camera height 0.88 m, agent radius 0.18 m

## Expert Data Collection (IL)

IL = Imitation Learning = clone expert behavior. The "expert" for ObjectNav is Habitat's built-in shortest-path oracle:

```python
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

follower = ShortestPathFollower(sim, goal_radius=0.2, return_one_hot=False)

# Collect expert demonstrations
episodes = []
for episode in env.episodes:
    obs = env.reset()
    trajectory = []
    while not env.episode_over:
        # Oracle computes optimal action using ground-truth navmesh
        expert_action = follower.get_next_action(env.current_episode.goals[0].position)
        trajectory.append({
            "rgb": obs["rgb"],              # [H, W, 3]
            "depth": obs["depth"],          # [H, W, 1]
            "goal_category": episode.object_category,  # "chair"
            "action": expert_action,        # 0-5 discrete
        })
        obs = env.step(expert_action)
    episodes.append(trajectory)
```

This gives (observation, expert_action) pairs. The policy learns to predict the same actions given the same observations — no reward function needed, just MSE/cross-entropy on expert actions.

**For discrete actions** (Phase 1): train a classification head on fusion output, cross-entropy loss against expert action labels.

**For continuous actions** (Phase 2): convert discrete actions to continuous velocities, then use diffusion/flow matching to predict action chunks.

## File Structure

```
src/action_heads/
├── __init__.py
├── common.py              # shared: SinusoidalPosEmb, MLP denoiser, EMA helper
├── ddpm_policy.py          # DDPMActionHead — ε-prediction denoising
├── flow_matching_policy.py # FlowMatchingActionHead — velocity prediction
├── noise_schedules.py      # linear, cosine β schedules + α̅ precomputation
├── conditioning.py         # FusionModule (concat, takes scene+goal cond)
└── discrete_head.py        # DiscreteActionHead — simple classifier baseline

scripts/
└── collect_expert_demos.py # ShortestPathFollower → dataset

notebooks/
└── action_heads_exploration.ipynb

tests/
└── test_action_heads.py
```

## API Design

All three heads share the same conditioning interface:

```python
# Phase 1: Discrete baseline (start here)
class DiscreteActionHead(nn.Module):
    def __init__(self, cond_dim=256, n_actions=6, hidden_dim=256):
        ...

    def forward(self, cond: Tensor) -> Tensor:
        """Returns action logits [B, n_actions]."""

    def sample(self, cond: Tensor) -> Tensor:
        """Returns argmax action [B]."""


# Phase 2: Continuous action heads (diffusion / flow matching)
class DDPMActionHead(nn.Module):
    def __init__(self, action_dim=4, cond_dim=256, chunk_size=4, n_steps=100,
                 hidden_dim=256, n_layers=4):
        ...

    def forward(self, cond: Tensor, noisy_actions: Tensor, timesteps: Tensor) -> Tensor:
        """Predict noise ε given (cond, noisy_actions, t). Used during training."""
        # returns predicted noise, same shape as noisy_actions

    @torch.no_grad()
    def sample(self, cond: Tensor, n_samples: int = 1) -> Tensor:
        """DDPM ancestral sampling: iterate T → 0, denoise from pure noise."""
        # returns [B, chunk_size, action_dim]


class FlowMatchingActionHead(nn.Module):
    def __init__(self, action_dim=4, cond_dim=256, chunk_size=4,
                 hidden_dim=256, n_layers=4, sigma_min=1e-4):
        ...

    def forward(self, cond: Tensor, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        """Predict velocity v given (cond, interpolated x_t, t). Used during training."""
        # returns predicted velocity, same shape as x_0

    @torch.no_grad()
    def sample(self, cond: Tensor, n_samples: int = 1, n_steps: int = 20) -> Tensor:
        """Euler ODE integration: x_{t+dt} = x_t + v(x_t, t) * dt."""
        # returns [B, chunk_size, action_dim]
```

### Shared MLP denoiser (inside both continuous heads)

```python
class MLPDenoiser(nn.Module):
    """MLP that predicts ε (DDPM) or v (FM) given concat(x_t, t_emb, cond)."""
    def __init__(self, action_dim, cond_dim, hidden_dim=256, n_layers=4):
        self.time_emb = SinusoidalPosEmb(hidden_dim)
        self.input_proj = nn.Linear(action_dim + hidden_dim + cond_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
            for _ in range(n_layers)
        ])
        self.out = nn.Linear(hidden_dim, action_dim)

    def forward(self, x_t, t, cond):
        t_emb = self.time_emb(t)
        h = torch.cat([x_t.flatten(1), t_emb, cond], dim=-1)
        h = self.input_proj(h)
        for layer in self.layers:
            h = layer(h) + h   # residual
        return self.out(h).unflatten(-1, (chunk_size, action_dim))
```

### EMA helper

```python
class EMA:
    def __init__(self, model, decay=0.999):
        self.shadow = {k: v.clone() for k, v in model.state_dict().items()}
        self.decay = decay

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v

    def apply(self, model):
        model.load_state_dict(self.shadow)
```

### FusionModule (conditioning)

```python
class FusionModule(nn.Module):
    """Fuses VGGT DPT outputs + goal into a single cond vector."""
    def __init__(self, clip_dim=768, spatial_channels=5, pose_dim=9, cond_dim=256):
        ...

    def forward(self, world_points, depth, relevance, pose_enc, goal_emb) -> Tensor:
        # returns [B, cond_dim]
```

## Training

### Phase 1: Discrete IL baseline

```python
for batch in dataloader:
    cond = fusion(batch["scene"], batch["goal"])
    logits = discrete_head(cond)                 # [B, 6]
    loss = F.cross_entropy(logits, batch["expert_action"])
    loss.backward()
    optimizer.step()
```

### Phase 2a: DDPM (continuous)

```python
ema = EMA(ddpm_head, decay=0.999)

for batch in dataloader:
    actions = batch["actions"]           # [B, chunk_size=4, action_dim]
    cond = fusion(batch["scene"], batch["goal"])

    t = torch.randint(0, n_steps, (B,))
    noise = torch.randn_like(actions)
    alpha_bar_t = alpha_bar[t]
    x_t = sqrt(alpha_bar_t) * actions + sqrt(1 - alpha_bar_t) * noise

    noise_pred = ddpm_head(cond, x_t, t)
    loss = F.mse_loss(noise_pred, noise)

    loss.backward()
    optimizer.step()
    ema.update(ddpm_head)
```

### Phase 2b: Flow Matching (continuous)

```python
ema = EMA(fm_head, decay=0.999)

for batch in dataloader:
    x_1 = batch["actions"]               # [B, chunk_size=4, action_dim]
    cond = fusion(batch["scene"], batch["goal"])

    x_0 = torch.randn_like(x_1)
    t = torch.rand(B, 1, 1)              # uniform [0, 1]

    x_t = (1 - t) * x_0 + t * x_1       # linear interpolation
    v_target = x_1 - x_0                 # optimal transport velocity

    v_pred = fm_head(cond, x_t, t.squeeze())
    loss = F.mse_loss(v_pred, v_target)

    loss.backward()
    optimizer.step()
    ema.update(fm_head)
```

### Key differences

| Aspect | DDPM | Flow Matching |
|--------|------|---------------|
| **Predicts** | noise ε added at step t | velocity v along OT path |
| **Time** | discrete t ∈ {0, ..., T-1}, T=100 | continuous t ∈ [0, 1] |
| **Forward** | Markov chain with β schedule | linear interp x_t = (1-t)x_0 + t·x_1 |
| **Sampling** | T denoising steps (slow) | Euler ODE, 10-20 steps (fast) |
| **Loss** | MSE(ε_pred, ε) | MSE(v_pred, x_1 - x_0) |
| **Inference** | ~100 forward passes | ~20 forward passes |
| **Multimodality** | Good | Better (straighter paths) |

## Notebook Outline: `action_heads_exploration.ipynb`

### Part 0: Expert Data
1. **ShortestPathFollower** — show oracle navigating to "chair" in HM3D
2. **Dataset structure** — (obs, action) pairs, action distribution histogram
3. **Discrete baseline** — train classifier, show accuracy

### Part 1: Diffusion Policy from Scratch
1. **Noise schedules** — visualize linear vs cosine β, plot α̅_t decay
2. **Forward process** — show 2D toy action being noised at t=0,25,50,75,100
3. **ε-prediction** — train tiny MLP on 2D Swiss Roll, visualize learned denoising
4. **Action chunks** — extend to [chunk_size=4, action_dim], show multi-step prediction
5. **Conditioned** — add goal conditioning, same noise → different actions for different goals

### Part 2: Flow Matching from Scratch
1. **OT paths** — visualize linear interpolation x_t = (1-t)x_0 + t·x_1
2. **Velocity field** — plot v = x_1 - x_0 as vector field on 2D plane
3. **CFM training** — train tiny MLP on same Swiss Roll, compare convergence to DDPM
4. **Euler sampling** — show ODE integration with varying n_steps (5, 10, 20, 50)
5. **Conditioned** — same goal-conditioning comparison

### Part 3: Head-to-Head Comparison
1. **Same toy task** — 2D navigation to goal point
2. **Training curves** — loss convergence side by side
3. **Sample quality** — overlay generated action trajectories
4. **Inference speed** — wall-clock time comparison
5. **Multimodality** — two valid paths to goal, which captures both modes?

### Part 4: Integration with FusionModule
1. **Dummy VGGT outputs** — synthetic depth, points, pose, goal_emb
2. **FusionModule forward** — show conditioning vector
3. **Full pipeline** — fusion → action head → sampled action chunk
4. **Ablation sketch** — remove relevance map, remove goal, observe effect

## Dependencies to Add

```toml
"tqdm",
```

No new heavy deps — both methods are pure PyTorch.

## Test Plan

```python
# test_action_heads.py

def test_discrete_head_shape():
    """DiscreteActionHead returns [B, n_actions=6] logits."""

def test_ddpm_forward_shape():
    """DDPMActionHead.forward returns [B, chunk=4, action_dim]."""

def test_ddpm_sample_shape():
    """DDPMActionHead.sample returns [B, chunk=4, action_dim]."""

def test_fm_forward_shape():
    """FlowMatchingActionHead.forward returns [B, chunk=4, action_dim]."""

def test_fm_sample_shape():
    """FlowMatchingActionHead.sample returns [B, chunk=4, action_dim]."""

def test_heads_swappable():
    """All heads accept same cond shape, produce compatible output."""

def test_fusion_module_shape():
    """FusionModule produces [B, cond_dim] from dummy VGGT outputs."""

def test_ddpm_noise_schedule():
    """α̅ monotonically decreasing, α̅_0 ≈ 1, α̅_T ≈ 0."""

def test_fm_interpolation():
    """x_t at t=0 equals x_0, at t=1 equals x_1."""

def test_ddpm_overfit_toy():
    """DDPM overfits single 2D action in <200 steps."""

def test_fm_overfit_toy():
    """FM overfits single 2D action in <200 steps."""

def test_ema_tracks_weights():
    """EMA shadow params move toward model params after update."""
```

## Phasing

| Phase | What | When |
|-------|------|------|
| **1** | `collect_expert_demos.py` + `DiscreteActionHead` + notebook Part 0 | First |
| **2** | `DDPMActionHead` + `FlowMatchingActionHead` + notebook Parts 1-3 | After discrete baseline works |
| **3** | `FusionModule` integration + notebook Part 4 | After VGGT pipeline ready |
| **4** | RL fine-tune via FMPG (optional) | After IL works end-to-end |
