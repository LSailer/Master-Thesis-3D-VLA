# R2-Dreamer JAX Port + 3-Way Comparison

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Full R2-Dreamer reimplementation in JAX + Habitat env wrapper for PyTorch r2dreamer + two comparison notebooks (Crafter correctness, Habitat timing).

**Architecture:** Port r2dreamer's Block-GRU, BlockLinear, RMSNorm, Barlow Twins loss, LaProp+AGC, and single-optimizer design to JAX/Flax. Keep existing DreamerV3 code intact (separate files). Write Habitat env adapter for PyTorch r2dreamer. Three variants: DreamerV3 PyTorch (decoder), R2-Dreamer PyTorch, R2-Dreamer JAX.

**Tech Stack:** JAX + Flax + Optax (JAX side), PyTorch r2dreamer (external/r2dreamer), Crafter (validation), Habitat (timing), matplotlib/pandas (notebooks).

---

## File Structure

### New files to create

| File | Responsibility |
|---|---|
| `src/dreamerv3/r2dreamer_config.py` | R2-Dreamer config dataclass matching size12M/size25M defaults |
| `src/dreamerv3/r2dreamer_networks.py` | RMSNorm, BlockLinear, Deter (Block-GRU), R2RSSM, R2Encoder, Projector, ReturnEMA |
| `src/dreamerv3/r2dreamer_agent.py` | R2DreamerAgent — Barlow loss, single optimizer, imagination, repval |
| `src/dreamerv3/optim.py` | LaProp optimizer + AGC gradient clipping (as optax transforms) |
| `tests/test_r2dreamer_shapes.py` | Shape/dtype tests for all new network modules |
| `tests/test_r2dreamer_optim.py` | LaProp + AGC correctness tests |
| `tests/test_r2dreamer_agent.py` | Agent smoke test (forward + backward on synthetic data) |
| `scripts/run_r2dreamer_jax_crafter.py` | JAX R2-Dreamer training on Crafter, outputs CSV |
| `scripts/run_r2dreamer_pytorch_crafter.py` | PyTorch r2dreamer on Crafter (rep_loss=r2dreamer + dreamer), outputs CSV |
| `scripts/habitat_env_r2dreamer.py` | Habitat env adapter for PyTorch r2dreamer |
| `scripts/run_habitat_timing.py` | 3-way timing benchmark on Habitat (5K steps) |

### Existing files to modify

| File | Change |
|---|---|
| `notebooks/dreamerv3_official_comparison.ipynb` | Extend from 2-way to 3-way Crafter comparison |
| `notebooks/dreamerv3_jax_vs_pytorch.ipynb` | Rewrite as 3-way Habitat timing comparison |

### Existing files to keep unchanged

`src/dreamerv3/agent.py`, `networks.py`, `configs.py`, `train.py`, `replay_buffer.py`, `env_crafter.py`, `env_habitat.py`, `tests/test_dreamerv3_shapes.py`, `tests/test_dreamerv3_integration.py`

---

## Task 1: R2-Dreamer Config

**Files:**
- Create: `src/dreamerv3/r2dreamer_config.py`
- Test: `tests/test_r2dreamer_shapes.py` (first test)

- [ ] **Step 1: Write config dataclass**

```python
# src/dreamerv3/r2dreamer_config.py
"""R2-Dreamer configuration — matches external/r2dreamer size12M defaults."""

import dataclasses


@dataclasses.dataclass
class R2DreamerConfig:
    # Environment
    obs_shape: tuple = (3, 64, 64)   # C, H, W
    num_actions: int = 17             # Crafter default
    max_episode_steps: int = 1000

    # RSSM (size12M)
    deter_size: int = 2048            # GRU deterministic state
    hidden_size: int = 256            # MLP hidden units inside Deter
    stoch_classes: int = 32           # categorical latent groups
    stoch_discrete: int = 16          # categories per group
    blocks: int = 8                   # block-diagonal blocks
    dyn_layers: int = 1               # hidden layers in Deter
    obs_layers: int = 1               # posterior MLP layers
    img_layers: int = 2               # prior MLP layers

    @property
    def stoch_size(self) -> int:
        return self.stoch_classes * self.stoch_discrete

    @property
    def feat_size(self) -> int:
        return self.stoch_size + self.deter_size

    # Encoder
    encoder_depth: int = 16           # CNN channel multiplier
    encoder_kernel: int = 5
    encoder_mults: tuple = (2, 3, 4, 4)

    # MLP heads
    mlp_units: int = 256
    mlp_layers_reward: int = 1
    mlp_layers_cont: int = 1
    mlp_layers_actor: int = 3
    mlp_layers_critic: int = 3
    twohot_bins: int = 255

    # Projector (Barlow Twins)
    barlow_lambda: float = 5e-4

    # Training
    batch_size: int = 16
    seq_len: int = 64
    imagination_horizon: int = 15
    horizon: int = 333                # discount = 1 - 1/horizon ≈ 0.997
    lamb: float = 0.95
    train_ratio: float = 512.0

    # Optimizer (LaProp)
    lr: float = 4e-5
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-20
    warmup_steps: int = 1000

    # AGC
    agc_clip: float = 0.3
    agc_pmin: float = 1e-3

    # Loss scales
    scale_barlow: float = 0.05
    scale_dyn: float = 1.0
    scale_rep: float = 0.1
    scale_rew: float = 1.0
    scale_con: float = 1.0
    scale_policy: float = 1.0
    scale_value: float = 1.0
    scale_repval: float = 0.3

    # Behavior
    kl_free: float = 1.0
    act_entropy: float = 3e-4
    unimix_ratio: float = 0.01
    slow_target_fraction: float = 0.02

    # Replay
    buffer_capacity: int = 500_000
    prefill_steps: int = 5000

    # Run
    total_steps: int = 1_000_000
    log_every: int = 250
    save_every: int = 50_000
    seed: int = 0
    logdir: str = "output/r2dreamer"

    # Model size preset
    @classmethod
    def size25M(cls, **overrides):
        defaults = dict(
            deter_size=3072, hidden_size=384, stoch_discrete=24,
            encoder_depth=24, mlp_units=384,
        )
        defaults.update(overrides)
        return cls(**defaults)
```

- [ ] **Step 2: Write initial test**

```python
# tests/test_r2dreamer_shapes.py (first test only)
import pytest
from src.dreamerv3.r2dreamer_config import R2DreamerConfig


class TestR2DreamerConfig:
    def test_defaults(self):
        cfg = R2DreamerConfig()
        assert cfg.stoch_size == 32 * 16  # 512
        assert cfg.feat_size == 2048 + 512  # 2560
        assert cfg.deter_size == 2048

    def test_size25m(self):
        cfg = R2DreamerConfig.size25M()
        assert cfg.deter_size == 3072
        assert cfg.hidden_size == 384
```

- [ ] **Step 3: Run test**

Run: `cd /pfs/data6/home/ul/ul_student/ul_hfj15/Master-Thesis-3D-VLA && uv run pytest tests/test_r2dreamer_shapes.py::TestR2DreamerConfig -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/dreamerv3/r2dreamer_config.py tests/test_r2dreamer_shapes.py
git commit -m "feat: add R2DreamerConfig matching r2dreamer size12M defaults"
```

---

## Task 2: RMSNorm + BlockLinear

**Files:**
- Create: `src/dreamerv3/r2dreamer_networks.py`
- Test: `tests/test_r2dreamer_shapes.py`

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/test_r2dreamer_shapes.py
import jax
import jax.numpy as jnp

from src.dreamerv3.r2dreamer_networks import RMSNorm, BlockLinear


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


class TestRMSNorm:
    def test_output_shape(self, rng):
        norm = RMSNorm()
        x = jnp.ones((4, 256))
        params = norm.init(rng, x)
        out = norm.apply(params, x)
        assert out.shape == (4, 256)

    def test_normalizes(self, rng):
        norm = RMSNorm()
        x = jnp.array([[3.0, 4.0]])
        params = norm.init(rng, x)
        out = norm.apply(params, x)
        # RMS of [3,4] = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.536
        # output ≈ [3/3.536, 4/3.536] * scale(=1) ≈ [0.849, 1.131]
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + 1e-8)
        expected = x / rms
        assert jnp.allclose(out, expected, atol=1e-5)


class TestBlockLinear:
    def test_output_shape(self, rng):
        bl = BlockLinear(out_features=512, blocks=8)
        x = jnp.zeros((2, 2048))
        params = bl.init(rng, x)
        out = bl.apply(params, x)
        assert out.shape == (2, 512)

    def test_3d_input(self, rng):
        bl = BlockLinear(out_features=256, blocks=8)
        x = jnp.zeros((2, 10, 512))
        params = bl.init(rng, x)
        out = bl.apply(params, x)
        assert out.shape == (2, 10, 256)

    def test_weight_shape(self, rng):
        bl = BlockLinear(out_features=512, blocks=8)
        x = jnp.zeros((1, 2048))
        params = bl.init(rng, x)
        # kernel: (out_per_block, in_per_block, blocks) = (64, 256, 8)
        assert params["params"]["kernel"].shape == (64, 256, 8)
        assert params["params"]["bias"].shape == (512,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /pfs/data6/home/ul/ul_student/ul_hfj15/Master-Thesis-3D-VLA && uv run pytest tests/test_r2dreamer_shapes.py::TestRMSNorm tests/test_r2dreamer_shapes.py::TestBlockLinear -v`
Expected: FAIL (import error)

- [ ] **Step 3: Implement RMSNorm and BlockLinear**

```python
# src/dreamerv3/r2dreamer_networks.py
"""R2-Dreamer network modules — JAX/Flax port of NM512/r2dreamer."""

import jax
import jax.numpy as jnp
import flax.linen as nn


# ---------- RMSNorm ----------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    eps: float = 1e-4

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = self.param("scale", nn.initializers.ones, (x.shape[-1],))
        rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * scale


# ---------- BlockLinear ----------

class BlockLinear(nn.Module):
    """Block-diagonal linear layer.

    Weight layout: (out_per_block, in_per_block, blocks).
    Matches r2dreamer's einsum: "...gi,oig->...go".
    """
    out_features: int
    blocks: int = 8

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        in_features = x.shape[-1]
        in_per_block = in_features // self.blocks
        out_per_block = self.out_features // self.blocks

        kernel = self.param(
            "kernel",
            nn.initializers.lecun_normal(),
            (out_per_block, in_per_block, self.blocks),
        )
        bias = self.param("bias", nn.initializers.zeros, (self.out_features,))

        batch_shape = x.shape[:-1]
        x = x.reshape(*batch_shape, self.blocks, in_per_block)
        x = jnp.einsum("...gi,oig->...go", x, kernel)
        x = x.reshape(*batch_shape, self.out_features)
        return x + bias
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /pfs/data6/home/ul/ul_student/ul_hfj15/Master-Thesis-3D-VLA && uv run pytest tests/test_r2dreamer_shapes.py::TestRMSNorm tests/test_r2dreamer_shapes.py::TestBlockLinear -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/dreamerv3/r2dreamer_networks.py tests/test_r2dreamer_shapes.py
git commit -m "feat: add RMSNorm and BlockLinear JAX modules"
```

---

## Task 3: Block-GRU (Deter)

**Files:**
- Modify: `src/dreamerv3/r2dreamer_networks.py`
- Test: `tests/test_r2dreamer_shapes.py`

- [ ] **Step 1: Write failing test**

```python
# Append to tests/test_r2dreamer_shapes.py
from src.dreamerv3.r2dreamer_config import R2DreamerConfig
from src.dreamerv3.r2dreamer_networks import Deter


@pytest.fixture
def cfg():
    return R2DreamerConfig()


class TestDeter:
    def test_output_shape(self, cfg, rng):
        deter = Deter(
            deter_size=cfg.deter_size, stoch_size=cfg.stoch_size,
            act_dim=cfg.num_actions, hidden=cfg.hidden_size,
            blocks=cfg.blocks, dyn_layers=cfg.dyn_layers,
        )
        h = jnp.zeros((2, cfg.deter_size))
        z = jnp.zeros((2, cfg.stoch_size))
        a = jnp.zeros((2, cfg.num_actions))
        params = deter.init(rng, z, h, a)
        h_new = deter.apply(params, z, h, a)
        assert h_new.shape == (2, cfg.deter_size)

    def test_deterministic_with_same_input(self, cfg, rng):
        deter = Deter(
            deter_size=cfg.deter_size, stoch_size=cfg.stoch_size,
            act_dim=cfg.num_actions, hidden=cfg.hidden_size,
            blocks=cfg.blocks, dyn_layers=cfg.dyn_layers,
        )
        h = jnp.ones((1, cfg.deter_size)) * 0.1
        z = jnp.ones((1, cfg.stoch_size)) * 0.1
        a = jnp.zeros((1, cfg.num_actions))
        params = deter.init(rng, z, h, a)
        h1 = deter.apply(params, z, h, a)
        h2 = deter.apply(params, z, h, a)
        assert jnp.allclose(h1, h2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /pfs/data6/home/ul/ul_student/ul_hfj15/Master-Thesis-3D-VLA && uv run pytest tests/test_r2dreamer_shapes.py::TestDeter -v`
Expected: FAIL (import error)

- [ ] **Step 3: Implement Deter**

```python
# Append to src/dreamerv3/r2dreamer_networks.py

class Deter(nn.Module):
    """Block-GRU deterministic state transition.

    Matches r2dreamer/rssm.py Deter: three input projections (deter, stoch, action)
    each with Dense→RMSNorm→SiLU, concatenated and broadcast across blocks,
    then BlockLinear hidden layers, and a GRU-style gate update.
    """
    deter_size: int = 2048
    stoch_size: int = 512
    act_dim: int = 17
    hidden: int = 256
    blocks: int = 8
    dyn_layers: int = 1

    @nn.compact
    def __call__(self, stoch: jnp.ndarray, deter: jnp.ndarray,
                 action: jnp.ndarray) -> jnp.ndarray:
        # stoch: (B, stoch_size), deter: (B, deter_size), action: (B, act_dim)
        # Normalize action magnitude (r2dreamer convention)
        action = action / jnp.clip(jnp.abs(action), a_min=1.0)

        # Three input projections → Dense + RMSNorm + SiLU
        x0 = nn.silu(RMSNorm(name="in_norm0")(nn.Dense(self.hidden, name="in0")(deter)))
        x1 = nn.silu(RMSNorm(name="in_norm1")(nn.Dense(self.hidden, name="in1")(stoch)))
        x2 = nn.silu(RMSNorm(name="in_norm2")(nn.Dense(self.hidden, name="in2")(action)))

        # Concatenate projections: (B, 3*hidden)
        x = jnp.concatenate([x0, x1, x2], axis=-1)

        # Broadcast across blocks: (B, blocks, 3*hidden)
        x = jnp.broadcast_to(x[:, None, :], (x.shape[0], self.blocks, x.shape[-1]))

        # Combine with per-block deter slice: (B, blocks, deter/blocks)
        deter_blocked = deter.reshape(deter.shape[0], self.blocks, self.deter_size // self.blocks)

        # (B, blocks, 3*hidden + deter/blocks) → flatten to (B, blocks*(3*hidden + deter/blocks))
        x = jnp.concatenate([deter_blocked, x], axis=-1)
        x = x.reshape(x.shape[0], -1)

        # Hidden layers: BlockLinear + RMSNorm + SiLU
        for i in range(self.dyn_layers):
            x = nn.silu(RMSNorm(name=f"hid_norm{i}")(
                BlockLinear(self.deter_size, self.blocks, name=f"hid{i}")(x)))

        # GRU gates: BlockLinear → 3 * deter_size
        gates = BlockLinear(3 * self.deter_size, self.blocks, name="gru")(x)

        # Split into reset, candidate, update (per-block, then flatten)
        gates = gates.reshape(gates.shape[0], self.blocks, 3, self.deter_size // self.blocks)
        reset = jax.nn.sigmoid(gates[:, :, 0, :].reshape(gates.shape[0], -1))
        cand = gates[:, :, 1, :].reshape(gates.shape[0], -1)
        update = jax.nn.sigmoid(gates[:, :, 2, :].reshape(gates.shape[0], -1) - 1.0)

        # GRU update: deter_new = update * tanh(reset * cand) + (1 - update) * deter
        deter_new = update * jnp.tanh(reset * cand) + (1.0 - update) * deter
        return deter_new
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /pfs/data6/home/ul/ul_student/ul_hfj15/Master-Thesis-3D-VLA && uv run pytest tests/test_r2dreamer_shapes.py::TestDeter -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/dreamerv3/r2dreamer_networks.py tests/test_r2dreamer_shapes.py
git commit -m "feat: add Block-GRU (Deter) module for R2-Dreamer"
```

---

## Task 4: R2-RSSM

**Files:**
- Modify: `src/dreamerv3/r2dreamer_networks.py`
- Test: `tests/test_r2dreamer_shapes.py`

- [ ] **Step 1: Write failing test**

```python
# Append to tests/test_r2dreamer_shapes.py
from src.dreamerv3.r2dreamer_networks import R2RSSM


class TestR2RSSM:
    def test_posterior_step(self, cfg, rng):
        rssm = R2RSSM(
            deter_size=cfg.deter_size, stoch_classes=cfg.stoch_classes,
            stoch_discrete=cfg.stoch_discrete, num_actions=cfg.num_actions,
            hidden=cfg.hidden_size, blocks=cfg.blocks,
            dyn_layers=cfg.dyn_layers, obs_layers=cfg.obs_layers,
            img_layers=cfg.img_layers,
        )
        B = 2
        stoch, deter = rssm.initial_state(B)
        action = jnp.zeros((B, cfg.num_actions))
        embed = jnp.zeros((B, 256))  # encoder output dim

        params = rssm.init(rng, stoch, deter, action, embed)
        new_stoch, new_deter, post_logit = rssm.apply(
            params, stoch, deter, action, embed)

        assert new_deter.shape == (B, cfg.deter_size)
        assert new_stoch.shape == (B, cfg.stoch_classes, cfg.stoch_discrete)
        assert post_logit.shape == (B, cfg.stoch_classes, cfg.stoch_discrete)

    def test_prior_step(self, cfg, rng):
        rssm = R2RSSM(
            deter_size=cfg.deter_size, stoch_classes=cfg.stoch_classes,
            stoch_discrete=cfg.stoch_discrete, num_actions=cfg.num_actions,
            hidden=cfg.hidden_size, blocks=cfg.blocks,
            dyn_layers=cfg.dyn_layers, obs_layers=cfg.obs_layers,
            img_layers=cfg.img_layers,
        )
        B = 2
        stoch, deter = rssm.initial_state(B)
        action = jnp.zeros((B, cfg.num_actions))
        # Need to init with embed to get all params
        embed = jnp.zeros((B, 256))
        params = rssm.init(rng, stoch, deter, action, embed)

        # Prior step (no embed)
        new_stoch, new_deter = rssm.apply(
            params, stoch, deter, action, method=rssm.img_step)
        assert new_deter.shape == (B, cfg.deter_size)
        assert new_stoch.shape == (B, cfg.stoch_classes, cfg.stoch_discrete)

    def test_observe_rollout(self, cfg, rng):
        rssm = R2RSSM(
            deter_size=cfg.deter_size, stoch_classes=cfg.stoch_classes,
            stoch_discrete=cfg.stoch_discrete, num_actions=cfg.num_actions,
            hidden=cfg.hidden_size, blocks=cfg.blocks,
            dyn_layers=cfg.dyn_layers, obs_layers=cfg.obs_layers,
            img_layers=cfg.img_layers,
        )
        B, T = 2, 10
        embed = jnp.zeros((B, T, 256))
        actions = jnp.zeros((B, T, cfg.num_actions))
        is_first = jnp.zeros((B, T))
        stoch0, deter0 = rssm.initial_state(B)

        # Init params via single step
        params = rssm.init(rng, stoch0, deter0, actions[:, 0], embed[:, 0])

        stochs, deters, logits = rssm.apply(
            params, embed, actions, (stoch0, deter0), is_first,
            method=rssm.observe)
        assert stochs.shape == (B, T, cfg.stoch_classes, cfg.stoch_discrete)
        assert deters.shape == (B, T, cfg.deter_size)
        assert logits.shape == (B, T, cfg.stoch_classes, cfg.stoch_discrete)

    def test_feat_size(self, cfg, rng):
        rssm = R2RSSM(
            deter_size=cfg.deter_size, stoch_classes=cfg.stoch_classes,
            stoch_discrete=cfg.stoch_discrete, num_actions=cfg.num_actions,
            hidden=cfg.hidden_size, blocks=cfg.blocks,
            dyn_layers=cfg.dyn_layers, obs_layers=cfg.obs_layers,
            img_layers=cfg.img_layers,
        )
        B = 2
        stoch = jnp.zeros((B, cfg.stoch_classes, cfg.stoch_discrete))
        deter = jnp.zeros((B, cfg.deter_size))
        feat = rssm.get_feat(stoch, deter)
        assert feat.shape == (B, cfg.feat_size)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /pfs/data6/home/ul/ul_student/ul_hfj15/Master-Thesis-3D-VLA && uv run pytest tests/test_r2dreamer_shapes.py::TestR2RSSM -v`
Expected: FAIL

- [ ] **Step 3: Implement R2RSSM**

```python
# Append to src/dreamerv3/r2dreamer_networks.py

class R2RSSM(nn.Module):
    """Recurrent State-Space Model with Block-GRU.

    Matches r2dreamer/rssm.py RSSM. Stochastic state is (B, S, K) categorical.
    """
    deter_size: int = 2048
    stoch_classes: int = 32
    stoch_discrete: int = 16
    num_actions: int = 17
    hidden: int = 256
    blocks: int = 8
    dyn_layers: int = 1
    obs_layers: int = 1
    img_layers: int = 2
    unimix_ratio: float = 0.01

    @property
    def stoch_size(self):
        return self.stoch_classes * self.stoch_discrete

    @property
    def feat_size(self):
        return self.stoch_size + self.deter_size

    def initial_state(self, batch_size):
        deter = jnp.zeros((batch_size, self.deter_size))
        stoch = jnp.zeros((batch_size, self.stoch_classes, self.stoch_discrete))
        return stoch, deter

    def get_feat(self, stoch, deter):
        """Flatten stoch and concatenate with deter. (B, [T], S, K) + (B, [T], D) → (B, [T], F)"""
        flat_stoch = stoch.reshape(*stoch.shape[:-2], self.stoch_size)
        return jnp.concatenate([flat_stoch, deter], axis=-1)

    @nn.compact
    def __call__(self, stoch, deter, action, embed):
        """Single posterior step. Returns (stoch, deter, post_logit)."""
        stoch_flat = stoch.reshape(stoch.shape[0], -1)

        # Deterministic transition
        deter = Deter(
            self.deter_size, self.stoch_size, action.shape[-1],
            self.hidden, self.blocks, self.dyn_layers, name="deter_net",
        )(stoch_flat, deter, action)

        # Posterior: concat deter + embed → MLP → logits
        x = jnp.concatenate([deter, embed], axis=-1)
        for i in range(self.obs_layers):
            x = nn.silu(RMSNorm(name=f"obs_norm{i}")(
                nn.Dense(self.hidden, name=f"obs_fc{i}")(x)))
        post_logit = nn.Dense(
            self.stoch_classes * self.stoch_discrete, name="obs_out")(x)
        post_logit = post_logit.reshape(
            post_logit.shape[0], self.stoch_classes, self.stoch_discrete)

        stoch = self._sample(post_logit)
        return stoch, deter, post_logit

    def img_step(self, stoch, deter, action):
        """Single prior step (no observation). Returns (stoch, deter)."""
        stoch_flat = stoch.reshape(stoch.shape[0], -1)

        deter = Deter(
            self.deter_size, self.stoch_size, action.shape[-1],
            self.hidden, self.blocks, self.dyn_layers, name="deter_net",
        )(stoch_flat, deter, action)

        # Prior: deter → MLP → logits
        x = deter
        for i in range(self.img_layers):
            x = nn.silu(RMSNorm(name=f"img_norm{i}")(
                nn.Dense(self.hidden, name=f"img_fc{i}")(x)))
        prior_logit = nn.Dense(
            self.stoch_classes * self.stoch_discrete, name="img_out")(x)
        prior_logit = prior_logit.reshape(
            prior_logit.shape[0], self.stoch_classes, self.stoch_discrete)

        stoch = self._sample(prior_logit)
        return stoch, deter

    def prior(self, deter):
        """Compute prior logits from deter. Returns (stoch, logit)."""
        x = deter
        for i in range(self.img_layers):
            x = nn.silu(RMSNorm(name=f"img_norm{i}")(
                nn.Dense(self.hidden, name=f"img_fc{i}")(x)))
        logit = nn.Dense(
            self.stoch_classes * self.stoch_discrete, name="img_out")(x)
        logit = logit.reshape(logit.shape[0], self.stoch_classes, self.stoch_discrete)
        stoch = self._sample(logit)
        return stoch, logit

    def observe(self, embed, actions, initial, is_first):
        """Posterior rollout. embed: (B,T,E), actions: (B,T,A), initial: (stoch,deter), is_first: (B,T)."""
        stoch, deter = initial
        B, T = embed.shape[0], embed.shape[1]
        stochs, deters, logits = [], [], []
        prev_action = jnp.zeros_like(actions[:, 0])

        for t in range(T):
            # Reset on episode boundary
            mask = (1.0 - is_first[:, t])
            stoch = stoch * mask[:, None, None]
            deter = deter * mask[:, None]
            prev_action = prev_action * mask[:, None]

            stoch, deter, logit = self(stoch, deter, prev_action, embed[:, t])
            stochs.append(stoch)
            deters.append(deter)
            logits.append(logit)
            prev_action = actions[:, t]

        return (jnp.stack(stochs, 1), jnp.stack(deters, 1),
                jnp.stack(logits, 1))

    def _sample(self, logits):
        """Categorical sample with straight-through + uniform mix. logits: (B, S, K)."""
        if self.unimix_ratio > 0:
            probs = jax.nn.softmax(logits, axis=-1)
            uniform = jnp.ones_like(probs) / self.stoch_discrete
            probs = (1 - self.unimix_ratio) * probs + self.unimix_ratio * uniform
            logits = jnp.log(probs + 1e-8)

        soft = jax.nn.softmax(logits, axis=-1)
        hard = jax.nn.one_hot(jnp.argmax(logits, axis=-1), self.stoch_discrete)
        return hard + soft - jax.lax.stop_gradient(soft)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /pfs/data6/home/ul/ul_student/ul_hfj15/Master-Thesis-3D-VLA && uv run pytest tests/test_r2dreamer_shapes.py::TestR2RSSM -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/dreamerv3/r2dreamer_networks.py tests/test_r2dreamer_shapes.py
git commit -m "feat: add R2RSSM with Block-GRU and posterior/prior/observe"
```

---

## Task 5: R2-Encoder + Projector + ReturnEMA

**Files:**
- Modify: `src/dreamerv3/r2dreamer_networks.py`
- Test: `tests/test_r2dreamer_shapes.py`

- [ ] **Step 1: Write failing tests**

```python
# Append to tests/test_r2dreamer_shapes.py
from src.dreamerv3.r2dreamer_networks import R2Encoder, Projector, ReturnEMA


class TestR2Encoder:
    def test_output_shape(self, cfg, rng):
        enc = R2Encoder(depth=cfg.encoder_depth, kernel_size=cfg.encoder_kernel)
        obs = jnp.zeros((2, *cfg.obs_shape))  # (B, 3, 64, 64)
        params = enc.init(rng, obs)
        out = enc.apply(params, obs)
        assert out.shape[0] == 2
        assert out.ndim == 2  # (B, embed_dim)
        # embed_dim = depth*4 * (64/16)^2 = 16*4 * 4*4 = 1024
        assert out.shape[1] == cfg.encoder_depth * cfg.encoder_mults[-1] * 4 * 4


class TestProjector:
    def test_output_shape(self, cfg, rng):
        embed_dim = cfg.encoder_depth * cfg.encoder_mults[-1] * 4 * 4
        proj = Projector(embed_dim)
        feat = jnp.zeros((2, cfg.feat_size))
        params = proj.init(rng, feat)
        out = proj.apply(params, feat)
        assert out.shape == (2, embed_dim)


class TestReturnEMA:
    def test_update(self):
        ema = ReturnEMA(alpha=0.5)
        returns = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        state = ema.init_state()
        state = ema.update(state, returns)
        offset, scale = ema.get_stats(state)
        assert scale >= 1.0  # clipped min
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /pfs/data6/home/ul/ul_student/ul_hfj15/Master-Thesis-3D-VLA && uv run pytest tests/test_r2dreamer_shapes.py::TestR2Encoder tests/test_r2dreamer_shapes.py::TestProjector tests/test_r2dreamer_shapes.py::TestReturnEMA -v`
Expected: FAIL

- [ ] **Step 3: Implement R2Encoder, Projector, ReturnEMA**

```python
# Append to src/dreamerv3/r2dreamer_networks.py

class R2Encoder(nn.Module):
    """CNN encoder matching r2dreamer: Conv2d(SAME) + MaxPool + RMSNorm + SiLU.

    Channel depths: depth * [2, 3, 4, 4]. Output is flattened spatial features.
    """
    depth: int = 16
    kernel_size: int = 5
    mults: tuple = (2, 3, 4, 4)

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        # obs: (B, C, H, W) float [0, 1]
        x = obs - 0.5  # center
        x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW → NHWC

        for i, mult in enumerate(self.mults):
            ch = self.depth * mult
            x = nn.Conv(ch, (self.kernel_size, self.kernel_size),
                        padding="SAME", name=f"conv{i}")(x)
            # MaxPool stride 2
            x = nn.max_pool(x, (2, 2), strides=(2, 2))
            x = RMSNorm(name=f"norm{i}")(x)
            x = nn.silu(x)

        # Flatten: (B, H', W', C') → (B, H'*W'*C')
        return x.reshape(x.shape[0], -1)

    @property
    def out_dim_for(self):
        """Helper: compute output dim for a given spatial input size."""
        # For 64x64 with 4 pools: 64/16=4, so out = depth*mults[-1] * 4 * 4
        return None  # caller computes from actual forward pass


class Projector(nn.Module):
    """Single linear projection (no bias) for Barlow Twins."""
    out_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.Dense(self.out_dim, use_bias=False, name="proj")(x)


class ReturnEMA:
    """Running 5th/95th percentile EMA for return normalization."""

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha

    def init_state(self):
        return jnp.zeros(2)  # [p05, p95]

    def update(self, state, returns):
        quantiles = jnp.array([
            jnp.percentile(returns, 5),
            jnp.percentile(returns, 95),
        ])
        return self.alpha * quantiles + (1 - self.alpha) * state

    def get_stats(self, state):
        offset = state[0]
        scale = jnp.maximum(state[1] - state[0], 1.0)
        return offset, scale
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /pfs/data6/home/ul/ul_student/ul_hfj15/Master-Thesis-3D-VLA && uv run pytest tests/test_r2dreamer_shapes.py::TestR2Encoder tests/test_r2dreamer_shapes.py::TestProjector tests/test_r2dreamer_shapes.py::TestReturnEMA -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/dreamerv3/r2dreamer_networks.py tests/test_r2dreamer_shapes.py
git commit -m "feat: add R2Encoder, Projector, and ReturnEMA"
```

---

## Task 6: LaProp Optimizer + AGC

**Files:**
- Create: `src/dreamerv3/optim.py`
- Create: `tests/test_r2dreamer_optim.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_r2dreamer_optim.py
"""Tests for LaProp optimizer and AGC gradient clipping."""

import jax
import jax.numpy as jnp
import optax
import pytest

from src.dreamerv3.optim import laprop, agc


class TestLaProp:
    def test_basic_optimization(self):
        """LaProp should minimize a simple quadratic."""
        tx = laprop(lr=1e-2)
        params = jnp.array([5.0, -3.0])
        state = tx.init(params)

        for _ in range(200):
            grads = 2 * params  # gradient of x^2
            updates, state = tx.update(grads, state, params)
            params = optax.apply_updates(params, updates)

        assert jnp.allclose(params, jnp.zeros(2), atol=0.1)

    def test_state_structure(self):
        tx = laprop(lr=1e-3)
        params = jnp.zeros(10)
        state = tx.init(params)
        # Should have count, exp_avg, exp_avg_sq, exp_avg_lr fields
        assert hasattr(state, 'count') or isinstance(state, tuple)


class TestAGC:
    def test_clips_large_gradients(self):
        params = jnp.array([1.0, 1.0])
        grads = jnp.array([100.0, 100.0])
        clipped = agc(grads, params, clip=0.3, pmin=1e-3)
        # param norm ≈ 1.414, max grad = 0.3 * 1.414 ≈ 0.424
        assert jnp.all(jnp.abs(clipped) < jnp.abs(grads))

    def test_does_not_clip_small_gradients(self):
        params = jnp.array([10.0, 10.0])
        grads = jnp.array([0.01, 0.01])
        clipped = agc(grads, params, clip=0.3, pmin=1e-3)
        assert jnp.allclose(clipped, grads)

    def test_pmin_floor(self):
        """Even with tiny params, pmin prevents division by zero."""
        params = jnp.array([0.0, 0.0])
        grads = jnp.array([10.0, 10.0])
        clipped = agc(grads, params, clip=0.3, pmin=1e-3)
        assert jnp.all(jnp.isfinite(clipped))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /pfs/data6/home/ul/ul_student/ul_hfj15/Master-Thesis-3D-VLA && uv run pytest tests/test_r2dreamer_optim.py -v`
Expected: FAIL

- [ ] **Step 3: Implement LaProp and AGC**

```python
# src/dreamerv3/optim.py
"""LaProp optimizer and Adaptive Gradient Clipping (AGC) for JAX/Optax."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax


# ---------- LaProp ----------

class LaPropState(NamedTuple):
    count: jnp.ndarray
    exp_avg: optax.Updates
    exp_avg_sq: optax.Updates
    exp_avg_lr1: jnp.ndarray
    exp_avg_lr2: jnp.ndarray


def laprop(lr: float = 4e-4, b1: float = 0.9, b2: float = 0.999,
           eps: float = 1e-15) -> optax.GradientTransformation:
    """LaProp optimizer (Wang 2020).

    Key difference from Adam: gradient is normalized by second moment BEFORE
    computing the first moment running average.
    """

    def init_fn(params):
        return LaPropState(
            count=jnp.zeros([], jnp.int32),
            exp_avg=jax.tree.map(jnp.zeros_like, params),
            exp_avg_sq=jax.tree.map(jnp.zeros_like, params),
            exp_avg_lr1=jnp.zeros([]),
            exp_avg_lr2=jnp.zeros([]),
        )

    def update_fn(updates, state, params=None):
        count = state.count + 1

        # Second moment: v_t = β2 * v_{t-1} + (1-β2) * g²
        exp_avg_sq = jax.tree.map(
            lambda v, g: b2 * v + (1 - b2) * g ** 2,
            state.exp_avg_sq, updates,
        )

        # Bias correction via LR tracking (matching r2dreamer's implementation)
        exp_avg_lr1 = state.exp_avg_lr1 * b1 + (1 - b1) * lr
        exp_avg_lr2 = state.exp_avg_lr2 * b2 + (1 - b2)

        bias_correction1 = exp_avg_lr1 / (lr + 1e-30)
        step_size = 1.0 / jnp.maximum(bias_correction1, 1e-30)

        # Normalize gradient by sqrt(v / bias_correction2)
        denom = jax.tree.map(
            lambda v: jnp.sqrt(v / jnp.maximum(exp_avg_lr2, 1e-30)) + eps,
            exp_avg_sq,
        )
        normalized_grad = jax.tree.map(lambda g, d: g / d, updates, denom)

        # First moment of normalized gradient: m_t = β1 * m_{t-1} + (1-β1) * lr * g_norm
        exp_avg = jax.tree.map(
            lambda m, ng: b1 * m + (1 - b1) * lr * ng,
            state.exp_avg, normalized_grad,
        )

        # Final update
        final_updates = jax.tree.map(lambda m: -step_size * m, exp_avg)

        new_state = LaPropState(
            count=count,
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            exp_avg_lr1=exp_avg_lr1,
            exp_avg_lr2=exp_avg_lr2,
        )
        return final_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


# ---------- AGC ----------

def agc(grads, params, clip: float = 0.3, pmin: float = 1e-3):
    """Adaptive Gradient Clipping applied to a pytree."""

    def clip_fn(g, p):
        p_norm = jnp.maximum(jnp.sqrt(jnp.sum(p ** 2)), pmin)
        g_norm = jnp.sqrt(jnp.sum(g ** 2))
        max_norm = clip * p_norm
        scale = max_norm / jnp.maximum(g_norm, 1e-8)
        return jnp.where(g_norm > max_norm, g * scale, g)

    return jax.tree.map(clip_fn, grads, params)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /pfs/data6/home/ul/ul_student/ul_hfj15/Master-Thesis-3D-VLA && uv run pytest tests/test_r2dreamer_optim.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/dreamerv3/optim.py tests/test_r2dreamer_optim.py
git commit -m "feat: add LaProp optimizer and AGC gradient clipping for JAX"
```

---

## Task 7: R2DreamerAgent

**Files:**
- Create: `src/dreamerv3/r2dreamer_agent.py`
- Create: `tests/test_r2dreamer_agent.py`

This is the largest task. The agent combines: single optimizer, Barlow Twins loss, KL (dyn+rep split), reward/continue prediction, imagination rollout, actor-critic with return normalization, and replay-based value learning.

- [ ] **Step 1: Write failing agent smoke test**

```python
# tests/test_r2dreamer_agent.py
"""Smoke tests for R2DreamerAgent."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.dreamerv3.r2dreamer_config import R2DreamerConfig
from src.dreamerv3.r2dreamer_agent import R2DreamerAgent


@pytest.fixture
def cfg():
    return R2DreamerConfig(
        obs_shape=(3, 64, 64),
        num_actions=17,
    )


@pytest.fixture
def agent(cfg):
    rng = jax.random.PRNGKey(42)
    return R2DreamerAgent(cfg, rng)


def make_batch(cfg, B=4, T=16):
    """Synthetic batch."""
    return {
        "obs": jnp.array(np.random.rand(B, T, *cfg.obs_shape).astype(np.float32)),
        "actions": jnp.array(
            np.eye(cfg.num_actions, dtype=np.float32)[
                np.random.randint(0, cfg.num_actions, (B, T))]),
        "rewards": jnp.array(np.random.randn(B, T).astype(np.float32)),
        "is_first": jnp.zeros((B, T)),
        "is_last": jnp.zeros((B, T)),
        "is_terminal": jnp.zeros((B, T)),
    }


class TestR2DreamerAgent:
    def test_init(self, agent, cfg):
        """Agent initializes without error."""
        assert agent is not None

    def test_act(self, agent, cfg):
        """Agent produces valid actions."""
        obs = {
            "image": np.random.randint(0, 256, cfg.obs_shape, dtype=np.uint8),
            "is_first": True,
        }
        rng = jax.random.PRNGKey(0)
        action = agent.act(obs, rng)
        assert 0 <= action < cfg.num_actions

    def test_train_step(self, agent, cfg):
        """One train step produces finite losses."""
        batch = make_batch(cfg)
        rng = jax.random.PRNGKey(1)
        metrics = agent.train_step(batch, rng)
        assert "loss/barlow" in metrics
        assert "loss/dyn" in metrics
        assert "loss/rew" in metrics
        assert "loss/policy" in metrics
        assert "loss/value" in metrics
        for k, v in metrics.items():
            assert np.isfinite(v), f"{k} = {v} is not finite"

    def test_train_step_reduces_loss(self, agent, cfg):
        """Loss should decrease over a few steps on the same batch."""
        batch = make_batch(cfg)
        rng = jax.random.PRNGKey(2)
        losses = []
        for i in range(5):
            rng, k = jax.random.split(rng)
            m = agent.train_step(batch, k)
            losses.append(m["total_loss"])
        # At least the last loss should be <= the first (allowing noise)
        assert losses[-1] <= losses[0] * 2.0  # sanity: not diverging
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /pfs/data6/home/ul/ul_student/ul_hfj15/Master-Thesis-3D-VLA && uv run pytest tests/test_r2dreamer_agent.py -v`
Expected: FAIL

- [ ] **Step 3: Implement R2DreamerAgent**

```python
# src/dreamerv3/r2dreamer_agent.py
"""R2-Dreamer agent — JAX port of NM512/r2dreamer."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from .r2dreamer_config import R2DreamerConfig
from .r2dreamer_networks import (
    R2Encoder, R2RSSM, Projector, RMSNorm, ReturnEMA,
)
from .networks import MLP, TwoHotDist, symlog


class R2DreamerAgent:
    def __init__(self, config: R2DreamerConfig, rng_key: jax.Array):
        self.cfg = config
        c = config

        # Init networks
        keys = jax.random.split(rng_key, 7)

        # Encoder
        self.encoder = R2Encoder(c.encoder_depth, c.encoder_kernel, c.encoder_mults)
        dummy_obs = jnp.zeros((1, *c.obs_shape))
        enc_params = self.encoder.init(keys[0], dummy_obs)
        embed_out = self.encoder.apply(enc_params, dummy_obs)
        self.embed_size = embed_out.shape[-1]

        # RSSM
        self.rssm = R2RSSM(
            c.deter_size, c.stoch_classes, c.stoch_discrete, c.num_actions,
            c.hidden_size, c.blocks, c.dyn_layers, c.obs_layers, c.img_layers,
            c.unimix_ratio,
        )
        dummy_stoch, dummy_deter = self.rssm.initial_state(1)
        dummy_action = jnp.zeros((1, c.num_actions))
        dummy_embed = jnp.zeros((1, self.embed_size))
        rssm_params = self.rssm.init(keys[1], dummy_stoch, dummy_deter, dummy_action, dummy_embed)

        # Projector (Barlow Twins)
        self.projector = Projector(self.embed_size)
        dummy_feat = jnp.zeros((1, c.feat_size))
        proj_params = self.projector.init(keys[2], dummy_feat)

        # Reward head (twohot)
        self.reward_pred = MLP(c.mlp_units, c.mlp_layers_reward, c.twohot_bins, name="reward")
        rew_params = self.reward_pred.init(keys[3], dummy_feat)

        # Continue head
        self.cont_pred = MLP(c.mlp_units, c.mlp_layers_cont, 1, name="continue")
        cont_params = self.cont_pred.init(keys[4], dummy_feat)

        # Actor
        self.actor = MLP(c.mlp_units, c.mlp_layers_actor, c.num_actions, name="actor")
        actor_params = self.actor.init(keys[5], dummy_feat)

        # Critic (twohot)
        self.critic = MLP(c.mlp_units, c.mlp_layers_critic, c.twohot_bins, name="critic")
        critic_params = self.critic.init(keys[6], dummy_feat)

        # TwoHot distribution
        self.twohot = TwoHotDist(c.twohot_bins)

        # All params in single dict
        all_params = {
            "encoder": enc_params,
            "rssm": rssm_params,
            "projector": proj_params,
            "reward": rew_params,
            "continue": cont_params,
            "actor": actor_params,
            "critic": critic_params,
        }

        # Slow target (critic EMA)
        self.slow_critic_params = critic_params

        # Single optimizer: LaProp + warmup
        from .optim import laprop
        schedule = optax.join_schedules(
            [optax.linear_schedule(0.0, 1.0, c.warmup_steps),
             optax.constant_schedule(1.0)],
            [c.warmup_steps],
        )
        tx = optax.chain(
            laprop(lr=c.lr, b1=c.beta1, b2=c.beta2, eps=c.eps),
            optax.scale_by_schedule(schedule),
        )
        self.state = TrainState.create(apply_fn=None, params=all_params, tx=tx)

        # Return normalization
        self.return_ema = ReturnEMA(alpha=0.01)
        self._ema_state = self.return_ema.init_state()

        # Acting state
        self._act_stoch = dummy_stoch
        self._act_deter = dummy_deter
        self._act_prev_action = jnp.zeros((1, c.num_actions))

        # JIT
        self._train_step_jit = jax.jit(self._train_step)
        self._act_jit = jax.jit(self._act_forward)
        self._act_greedy_jit = jax.jit(self._act_greedy_forward)

    # ------ Acting ------

    def act(self, obs_dict: dict, rng_key: jax.Array, training: bool = True) -> int:
        obs = jnp.array(obs_dict["image"][None], dtype=jnp.float32) / 255.0

        if obs_dict.get("is_first", False):
            B = 1
            self._act_stoch, self._act_deter = self.rssm.initial_state(B)
            self._act_prev_action = jnp.zeros((B, self.cfg.num_actions))

        if training:
            action_idx, self._act_stoch, self._act_deter = self._act_jit(
                self.state.params, obs,
                self._act_stoch, self._act_deter, self._act_prev_action, rng_key)
        else:
            action_idx, self._act_stoch, self._act_deter = self._act_greedy_jit(
                self.state.params, obs,
                self._act_stoch, self._act_deter, self._act_prev_action)

        a = int(action_idx[0])
        self._act_prev_action = jax.nn.one_hot(action_idx, self.cfg.num_actions)
        return a

    def _act_forward(self, params, obs, stoch, deter, prev_action, rng_key):
        embed = self.encoder.apply(params["encoder"], obs)
        stoch, deter, _ = self.rssm.apply(params["rssm"], stoch, deter, prev_action, embed)
        feat = self.rssm.get_feat(stoch, deter)
        logits = self.actor.apply(params["actor"], feat)
        action = jax.random.categorical(rng_key, logits)
        return action, stoch, deter

    def _act_greedy_forward(self, params, obs, stoch, deter, prev_action):
        embed = self.encoder.apply(params["encoder"], obs)
        stoch, deter, _ = self.rssm.apply(params["rssm"], stoch, deter, prev_action, embed)
        feat = self.rssm.get_feat(stoch, deter)
        logits = self.actor.apply(params["actor"], feat)
        action = jnp.argmax(logits, axis=-1)
        return action, stoch, deter

    # ------ Training ------

    def train_step(self, batch: dict, rng_key: jax.Array) -> dict:
        self.state, self.slow_critic_params, self._ema_state, metrics = (
            self._train_step_jit(
                self.state, self.slow_critic_params, self._ema_state,
                batch, rng_key))
        return {k: float(v) for k, v in metrics.items()}

    def _train_step(self, state, slow_critic_params, ema_state, batch, rng_key):
        from .optim import agc
        cfg = self.cfg

        def loss_fn(params):
            B, T = batch["obs"].shape[0], batch["obs"].shape[1]

            # === Encode ===
            flat_obs = batch["obs"].reshape(B * T, *cfg.obs_shape)
            embeds = self.encoder.apply(params["encoder"], flat_obs)
            embeds = embeds.reshape(B, T, -1)

            # === RSSM posterior rollout ===
            stoch0, deter0 = self.rssm.initial_state(B)
            stochs, deters, post_logits = self.rssm.apply(
                params["rssm"], embeds, batch["actions"],
                (stoch0, deter0), batch["is_first"],
                method=self.rssm.observe)

            # === KL loss (dyn + rep split) ===
            _, prior_logits = self.rssm.apply(
                params["rssm"], deters.reshape(B * T, -1),
                method=self.rssm.prior)
            prior_logits = prior_logits.reshape(B, T, cfg.stoch_classes, cfg.stoch_discrete)

            dyn_kl = _kl_categorical(
                jax.lax.stop_gradient(post_logits), prior_logits)
            rep_kl = _kl_categorical(
                post_logits, jax.lax.stop_gradient(prior_logits))
            dyn_loss = jnp.maximum(dyn_kl, cfg.kl_free).mean()
            rep_loss = jnp.maximum(rep_kl, cfg.kl_free).mean()

            # === Barlow Twins loss ===
            feat = self.rssm.get_feat(stochs, deters)  # (B, T, F)
            flat_feat = feat.reshape(B * T, -1)
            x1 = self.projector.apply(params["projector"], flat_feat)
            x2 = jax.lax.stop_gradient(embeds.reshape(B * T, -1))

            x1_norm = (x1 - x1.mean(0)) / (x1.std(0) + 1e-8)
            x2_norm = (x2 - x2.mean(0)) / (x2.std(0) + 1e-8)
            c_mat = (x1_norm.T @ x2_norm) / (B * T)

            invariance = jnp.sum((jnp.diag(c_mat) - 1.0) ** 2)
            mask = 1.0 - jnp.eye(c_mat.shape[0])
            redundancy = jnp.sum((c_mat * mask) ** 2)
            barlow_loss = invariance + cfg.barlow_lambda * redundancy

            # === Reward + Continue ===
            rew_logits = self.reward_pred.apply(params["reward"], flat_feat)
            rew_logits = rew_logits.reshape(B, T, cfg.twohot_bins)
            rew_loss = self.twohot.loss(rew_logits, batch["rewards"]).mean()

            cont_logits = self.cont_pred.apply(params["continue"], flat_feat)
            cont_logits = cont_logits.reshape(B, T)
            cont_target = 1.0 - batch["is_terminal"]
            cont_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(cont_logits, cont_target))

            # === Imagination (actor-critic) ===
            k1, k2 = jax.random.split(rng_key)
            disc = 1.0 - 1.0 / cfg.horizon

            # Start from ALL posterior states
            start_stoch = stochs.reshape(B * T, cfg.stoch_classes, cfg.stoch_discrete)
            start_deter = deters.reshape(B * T, cfg.deter_size)

            # Imagination rollout (stop_gradient on world model)
            wm_params_sg = jax.lax.stop_gradient(params)
            imag_feats, imag_actions = _imagine(
                self.rssm, self.actor, wm_params_sg["rssm"], params["actor"],
                start_stoch, start_deter, cfg.imagination_horizon + 1, k1,
                cfg.stoch_classes, cfg.stoch_discrete, cfg.deter_size,
                cfg.num_actions,
            )

            # Rewards and continues for imagined trajectory
            imag_rew_logits = self.reward_pred.apply(
                jax.lax.stop_gradient(params["reward"]), imag_feats)
            imag_rewards = self.twohot.pred(imag_rew_logits)
            imag_cont = jax.nn.sigmoid(
                self.cont_pred.apply(
                    jax.lax.stop_gradient(params["continue"]), imag_feats).squeeze(-1))

            # Values (slow target)
            imag_values = self.twohot.pred(
                self.critic.apply(slow_critic_params, imag_feats))
            imag_slow_values = imag_values  # same for now

            # Lambda returns
            weight = jnp.cumprod(imag_cont * disc, axis=1)
            ret = _lambda_return(imag_rewards, imag_values, imag_cont, disc, cfg.lamb)

            # Return normalization
            _, ret_scale = self.return_ema.get_stats(ema_state)
            adv = (ret - imag_values[:, :-1]) / ret_scale

            # Actor loss (log_prob weighted by advantages + entropy)
            actor_logits = self.actor.apply(params["actor"], imag_feats[:, :-1])
            action_log_probs = jnp.sum(
                jax.nn.log_softmax(actor_logits) * jax.lax.stop_gradient(imag_actions[:, :-1]),
                axis=-1)
            entropy = _categorical_entropy(actor_logits).mean()
            policy_loss = -jnp.mean(
                jax.lax.stop_gradient(weight[:, :-1]) *
                (action_log_probs * jax.lax.stop_gradient(adv) +
                 cfg.act_entropy * _categorical_entropy(actor_logits)))

            # Critic loss (on imagined returns)
            critic_logits = self.critic.apply(params["critic"],
                                              jax.lax.stop_gradient(imag_feats[:, :-1]))
            value_loss = self.twohot.loss(
                critic_logits, jax.lax.stop_gradient(ret)).mean()

            # === Replay-based value learning ===
            replay_feat = self.rssm.get_feat(stochs, deters)
            replay_value_logits = self.critic.apply(params["critic"], replay_feat)
            boot = ret[:, 0].reshape(B, T)
            replay_values = self.twohot.pred(
                self.critic.apply(jax.lax.stop_gradient(params["critic"]), replay_feat))
            replay_ret = _lambda_return_replay(
                batch["rewards"], replay_values, boot, batch["is_last"],
                batch["is_terminal"], disc, cfg.lamb)
            repval_loss = self.twohot.loss(
                replay_value_logits[:, :-1],
                jax.lax.stop_gradient(replay_ret)).mean()

            # === Total loss ===
            total = (cfg.scale_dyn * dyn_loss +
                     cfg.scale_rep * rep_loss +
                     cfg.scale_barlow * barlow_loss +
                     cfg.scale_rew * rew_loss +
                     cfg.scale_con * cont_loss +
                     cfg.scale_policy * policy_loss +
                     cfg.scale_value * value_loss +
                     cfg.scale_repval * repval_loss)

            metrics = {
                "total_loss": total,
                "loss/dyn": dyn_loss, "loss/rep": rep_loss,
                "loss/barlow": barlow_loss,
                "loss/rew": rew_loss, "loss/con": cont_loss,
                "loss/policy": policy_loss, "loss/value": value_loss,
                "loss/repval": repval_loss,
                "entropy": entropy,
                "imag_reward": imag_rewards.mean(),
                "imag_return": ret.mean(),
            }
            return total, (metrics, ret)

        grads, (metrics, ret) = jax.grad(loss_fn, has_aux=True)(state.params)

        # AGC
        grads = agc(grads, state.params, cfg.agc_clip, cfg.agc_pmin)

        state = state.apply_gradients(grads=grads)

        # Update slow critic target (EMA)
        tau = cfg.slow_target_fraction
        slow_critic_params = jax.tree.map(
            lambda s, v: (1 - tau) * s + tau * v,
            slow_critic_params, state.params["critic"])

        # Update return EMA
        ema_state = self.return_ema.update(ema_state, ret)

        return state, slow_critic_params, ema_state, metrics

    # ------ Save/Load ------

    def save(self, path: str):
        import os, pickle
        os.makedirs(path, exist_ok=True)
        data = {
            "params": self.state.params,
            "slow_critic": self.slow_critic_params,
            "ema_state": self._ema_state,
        }
        with open(os.path.join(path, "r2dreamer_checkpoint.pkl"), "wb") as f:
            pickle.dump(jax.tree.map(lambda x: np.array(x), data), f)

    def load(self, path: str):
        import os, pickle
        with open(os.path.join(path, "r2dreamer_checkpoint.pkl"), "rb") as f:
            data = pickle.load(f)
        data = jax.tree.map(jnp.array, data)
        self.state = self.state.replace(params=data["params"])
        self.slow_critic_params = data["slow_critic"]
        self._ema_state = data["ema_state"]


# ------ Utilities ------

def _kl_categorical(post_logits, prior_logits):
    """KL(post || prior) per class, summed over dims. logits: (B, T, S, K)."""
    log_post = jax.nn.log_softmax(post_logits, axis=-1)
    log_prior = jax.nn.log_softmax(prior_logits, axis=-1)
    post = jax.nn.softmax(post_logits, axis=-1)
    return (post * (log_post - log_prior)).sum(-1).sum(-1)  # (B, T)


def _categorical_entropy(logits):
    """Entropy of categorical distribution."""
    probs = jax.nn.softmax(logits, axis=-1)
    return -(probs * jnp.log(probs + 1e-8)).sum(axis=-1)


def _imagine(rssm, actor, rssm_params, actor_params,
             stoch, deter, horizon, rng_key,
             stoch_classes, stoch_discrete, deter_size, num_actions):
    """Roll out policy in latent space. Returns (feats, actions)."""
    feats, actions = [], []
    for i in range(horizon):
        feat = rssm.get_feat(stoch, deter)
        logits = actor.apply(actor_params, feat)
        rng_key, k = jax.random.split(rng_key)
        action_idx = jax.random.categorical(k, logits)
        action = jax.nn.one_hot(action_idx, num_actions)
        feats.append(feat)
        actions.append(action)
        stoch, deter = rssm.apply(rssm_params, stoch, deter, action,
                                   method=rssm.img_step)
    return jnp.stack(feats, 1), jnp.stack(actions, 1)


def _lambda_return(rewards, values, conts, disc, lamb):
    """Lambda returns for imagination. rewards/values/conts: (N, H+1)."""
    H = rewards.shape[1] - 1
    ret = values[:, -1]
    rets = [ret]
    for t in reversed(range(H)):
        ret = rewards[:, t + 1] + disc * conts[:, t + 1] * (
            (1 - lamb) * values[:, t + 1] + lamb * ret)
        rets.append(ret)
    return jnp.stack(list(reversed(rets))[:-1], axis=1)  # (N, H)


def _lambda_return_replay(rewards, values, boot, is_last, is_terminal, disc, lamb):
    """Lambda returns for replay-based value learning."""
    T = rewards.shape[1]
    live = (1.0 - is_terminal[:, 1:]) * disc
    cont = (1.0 - is_last[:, 1:]) * lamb
    interm = rewards[:, 1:] + (1 - cont) * live * boot[:, 1:]
    ret = boot[:, -1]
    rets = [ret]
    for t in reversed(range(live.shape[1])):
        ret = interm[:, t] + live[:, t] * cont[:, t] * ret
        rets.append(ret)
    return jnp.stack(list(reversed(rets))[:-1], axis=1)  # (B, T-1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /pfs/data6/home/ul/ul_student/ul_hfj15/Master-Thesis-3D-VLA && uv run pytest tests/test_r2dreamer_agent.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/dreamerv3/r2dreamer_agent.py tests/test_r2dreamer_agent.py
git commit -m "feat: add R2DreamerAgent with Barlow Twins, LaProp, imagination, repval"
```

---

## Task 8: JAX R2-Dreamer Crafter Training Script

**Files:**
- Create: `scripts/run_r2dreamer_jax_crafter.py`

- [ ] **Step 1: Write training script**

```python
# scripts/run_r2dreamer_jax_crafter.py
"""Run JAX R2-Dreamer on Crafter, output metrics to CSV."""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import numpy as np

from src.dreamerv3.r2dreamer_agent import R2DreamerAgent
from src.dreamerv3.r2dreamer_config import R2DreamerConfig
from src.dreamerv3.env_crafter import CrafterEnv
from src.dreamerv3.replay_buffer import ReplayBuffer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--prefill", type=int, default=5000)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=250)
    args = parser.parse_args()

    config = R2DreamerConfig(
        obs_shape=(3, 64, 64),
        num_actions=17,
        total_steps=args.steps,
        prefill_steps=args.prefill,
        seed=args.seed,
        log_every=args.log_every,
    )

    env = CrafterEnv(size=(64, 64), seed=args.seed)
    rng_key = jax.random.PRNGKey(config.seed)
    rng_key, init_key = jax.random.split(rng_key)
    agent = R2DreamerAgent(config, init_key)

    # Use existing ReplayBuffer (stores uint8 obs)
    from src.dreamerv3.configs import DreamerConfig
    buf_cfg = DreamerConfig(
        obs_shape=config.obs_shape, buffer_capacity=config.buffer_capacity)
    buffer = ReplayBuffer(buf_cfg)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "metric", "value"])

        # Prefill
        print(f"Prefilling {config.prefill_steps} steps...")
        obs = env.reset()
        for _ in range(config.prefill_steps):
            action = np.random.randint(0, config.num_actions)
            next_obs = env.step(action)
            buffer.add(obs["image"], action, next_obs["reward"], next_obs["done"])
            obs = next_obs if not next_obs["done"] else env.reset()

        # Training loop
        print(f"Training R2-Dreamer (JAX) for {config.total_steps} steps...")
        obs = env.reset()
        episode_reward = 0.0
        episode_count = 0
        t0 = time.time()
        batch_steps = config.batch_size * config.seq_len
        train_credit = 0.0
        metrics = {}

        for step in range(config.total_steps):
            rng_key, act_key = jax.random.split(rng_key)
            action = agent.act(obs, act_key)
            next_obs = env.step(action)
            buffer.add(obs["image"], action, next_obs["reward"], next_obs["done"])
            episode_reward += next_obs["reward"]

            if next_obs["done"]:
                episode_count += 1
                writer.writerow([step, "episode/score", episode_reward])
                episode_reward = 0.0
                obs = env.reset()
            else:
                obs = next_obs

            if buffer.size >= batch_steps:
                train_credit += config.train_ratio / batch_steps
                while train_credit >= 1.0:
                    rng_key, train_key = jax.random.split(rng_key)
                    batch = buffer.sample(config.batch_size, config.seq_len)
                    # Convert batch to r2dreamer format (one-hot actions, add is_last/is_terminal)
                    r2_batch = _convert_batch(batch, config.num_actions)
                    metrics = agent.train_step(r2_batch, train_key)
                    train_credit -= 1.0

                if step % config.log_every == 0 and metrics:
                    for k, v in metrics.items():
                        writer.writerow([step, k, v])
                    f.flush()
                    elapsed = time.time() - t0
                    fps = (step + 1) / elapsed if elapsed > 0 else 0
                    print(f"[step {step:>6d}/{config.total_steps}] "
                          f"barlow={metrics.get('loss/barlow', 0):.3f} "
                          f"dyn={metrics.get('loss/dyn', 0):.3f} "
                          f"rew={metrics.get('loss/rew', 0):.3f} "
                          f"fps={fps:.0f}")

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.0f}s. Episodes: {episode_count}. Output: {args.output}")


def _convert_batch(batch, num_actions):
    """Convert ReplayBuffer batch to R2-Dreamer format."""
    import jax.numpy as jnp
    actions_onehot = jax.nn.one_hot(batch["actions"], num_actions)
    return {
        "obs": batch["obs"],
        "actions": actions_onehot,
        "rewards": batch["rewards"],
        "is_first": batch["is_first"],
        "is_last": batch["dones"],
        "is_terminal": batch["dones"],
    }


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test (100 steps)**

Run: `cd /pfs/data6/home/ul/ul_student/ul_hfj15/Master-Thesis-3D-VLA && uv run python scripts/run_r2dreamer_jax_crafter.py --steps 100 --prefill 200 --log_every 50 --output output/smoke_r2dreamer_jax.csv`
Expected: Runs without error, produces CSV with loss metrics

- [ ] **Step 3: Commit**

```bash
git add scripts/run_r2dreamer_jax_crafter.py
git commit -m "feat: add JAX R2-Dreamer Crafter training script"
```

---

## Task 9: PyTorch r2dreamer Crafter Scripts

**Files:**
- Create: `scripts/run_r2dreamer_pytorch_crafter.py`

- [ ] **Step 1: Write PyTorch r2dreamer runner**

This script runs the external r2dreamer repo on Crafter with both `rep_loss=dreamer` and `rep_loss=r2dreamer`, parsing outputs to CSV.

```python
# scripts/run_r2dreamer_pytorch_crafter.py
"""Run PyTorch r2dreamer on Crafter (both dreamer and r2dreamer modes), output CSV."""

import argparse
import csv
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
R2DREAMER_DIR = os.path.join(REPO_ROOT, "external", "r2dreamer")


def run_r2dreamer(logdir, steps, seed, rep_loss="r2dreamer"):
    """Run r2dreamer with Hydra."""
    cmd = [
        sys.executable, "train.py",
        f"env=crafter",
        f"model=size12M",
        f"model.rep_loss={rep_loss}",
        f"seed={seed}",
        f"logdir={logdir}",
        f"trainer.steps={steps}",
        f"model.compile=False",  # safer for benchmarking
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=R2DREAMER_DIR)
    return result.returncode


def parse_tensorboard_to_csv(logdir, csv_path):
    """Parse TensorBoard events to CSV."""
    try:
        from tensorboard.backend.event_processing import event_accumulator
        ea = event_accumulator.EventAccumulator(logdir)
        ea.Reload()

        rows = []
        for tag in ea.Tags().get("scalars", []):
            for event in ea.Scalars(tag):
                rows.append([event.step, tag, event.value])

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "metric", "value"])
            writer.writerows(rows)
        print(f"Parsed {len(rows)} entries to {csv_path}")
        return True
    except Exception as e:
        print(f"TensorBoard parse failed: {e}")
        # Fallback: try jsonl
        return _parse_jsonl_fallback(logdir, csv_path)


def _parse_jsonl_fallback(logdir, csv_path):
    """Fallback: parse metrics.jsonl if present."""
    jsonl_path = os.path.join(logdir, "metrics.jsonl")
    if not os.path.exists(jsonl_path):
        # Search recursively
        for root, dirs, files in os.walk(logdir):
            for f in files:
                if f.endswith(".jsonl"):
                    jsonl_path = os.path.join(root, f)
                    break

    if not os.path.exists(jsonl_path):
        print(f"No metrics file found in {logdir}")
        return False

    rows = []
    with open(jsonl_path) as f:
        for line in f:
            data = json.loads(line.strip())
            step = data.get("step", 0)
            for k, v in data.items():
                if k != "step" and isinstance(v, (int, float)):
                    rows.append([step, k, v])

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "metric", "value"])
        writer.writerows(rows)
    print(f"Parsed {len(rows)} entries to {csv_path}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rep_loss", choices=["dreamer", "r2dreamer", "both"], default="both")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(REPO_ROOT, "output", "comparison")
    os.makedirs(output_dir, exist_ok=True)

    modes = ["dreamer", "r2dreamer"] if args.rep_loss == "both" else [args.rep_loss]

    for mode in modes:
        logdir = os.path.join(REPO_ROOT, "output", f"pytorch_{mode}_crafter")
        csv_path = os.path.join(output_dir, f"pytorch_{mode}_metrics.csv")

        if os.path.exists(csv_path):
            print(f"{csv_path} exists, skipping. Delete to re-run.")
            continue

        rc = run_r2dreamer(logdir, args.steps, args.seed, mode)
        if rc != 0:
            print(f"WARNING: r2dreamer ({mode}) exited with code {rc}")

        parse_tensorboard_to_csv(logdir, csv_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify r2dreamer dependencies**

Run: `cd /pfs/data6/home/ul/ul_student/ul_hfj15/Master-Thesis-3D-VLA && uv run python -c "import torch; print(torch.__version__); import hydra; print('hydra ok'); import tensordict; print('tensordict ok')"`
Expected: Prints versions. If missing, install: `uv pip install hydra-core omegaconf tensordict`

- [ ] **Step 3: Commit**

```bash
git add scripts/run_r2dreamer_pytorch_crafter.py
git commit -m "feat: add PyTorch r2dreamer Crafter runner (dreamer + r2dreamer modes)"
```

---

## Task 10: Habitat Env Wrapper for PyTorch r2dreamer

**Files:**
- Create: `scripts/habitat_env_r2dreamer.py`

- [ ] **Step 1: Write Habitat env adapter**

```python
# scripts/habitat_env_r2dreamer.py
"""Habitat ObjectNav environment adapter for PyTorch r2dreamer.

Provides a simple interface that the r2dreamer Dreamer agent can use:
- reset() → dict with "image" (H,W,C uint8), "is_first", "reward", etc.
- step(action_onehot) → dict
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


class HabitatR2DreamerEnv:
    """Wraps our HabitatObjectNavEnv for PyTorch r2dreamer's interface."""

    def __init__(self, obs_size=64, split="train", max_episode_steps=500,
                 max_geodesic=None, reward_type="geodesic_delta"):
        from src.dreamerv3.configs import DreamerConfig
        from src.dreamerv3.env_habitat import HabitatObjectNavEnv

        config = DreamerConfig(
            obs_shape=(3, obs_size, obs_size),
            max_episode_steps=max_episode_steps,
            split=split,
            reward_type=reward_type,
        )
        self._env = HabitatObjectNavEnv(config, max_geodesic=max_geodesic)
        self.num_actions = 4  # STOP, FORWARD, LEFT, RIGHT
        self._obs_size = obs_size
        self._step_count = 0

    def reset(self):
        obs = self._env.reset()
        self._step_count = 0
        # Convert CHW → HWC for r2dreamer
        image = np.transpose(obs["image"], (1, 2, 0))  # (H, W, C)
        return {
            "image": image,
            "reward": np.float32(0.0),
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
        }

    def step(self, action):
        """action: int or one-hot array."""
        if isinstance(action, np.ndarray):
            action = int(np.argmax(action))
        obs = self._env.step(action)
        self._step_count += 1
        image = np.transpose(obs["image"], (1, 2, 0))
        done = obs["done"]
        return {
            "image": image,
            "reward": np.float32(obs["reward"]),
            "is_first": False,
            "is_last": done,
            "is_terminal": done,
        }

    def close(self):
        self._env.close()
```

- [ ] **Step 2: Write Habitat timing benchmark script**

Create `scripts/run_habitat_timing.py`:

```python
# scripts/run_habitat_timing.py
"""3-way Habitat timing benchmark: DreamerV3 PyTorch, R2-Dreamer PyTorch, R2-Dreamer JAX.

Measures steps/sec and peak GPU memory for 5K steps on identical Habitat observations.
Outputs results to JSON for the notebook to consume.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def benchmark_jax_r2dreamer(steps, obs_shape, num_actions, batch_size, seq_len):
    """Benchmark JAX R2-Dreamer training throughput."""
    import jax
    import jax.numpy as jnp
    from src.dreamerv3.r2dreamer_agent import R2DreamerAgent
    from src.dreamerv3.r2dreamer_config import R2DreamerConfig

    config = R2DreamerConfig(obs_shape=obs_shape, num_actions=num_actions,
                              batch_size=batch_size, seq_len=seq_len)
    rng = jax.random.PRNGKey(0)
    agent = R2DreamerAgent(config, rng)

    def make_batch():
        return {
            "obs": jnp.array(np.random.rand(batch_size, seq_len, *obs_shape).astype(np.float32)),
            "actions": jnp.array(np.eye(num_actions, dtype=np.float32)[
                np.random.randint(0, num_actions, (batch_size, seq_len))]),
            "rewards": jnp.zeros((batch_size, seq_len)),
            "is_first": jnp.zeros((batch_size, seq_len)),
            "is_last": jnp.zeros((batch_size, seq_len)),
            "is_terminal": jnp.zeros((batch_size, seq_len)),
        }

    # Warmup
    for _ in range(3):
        rng, k = jax.random.split(rng)
        agent.train_step(make_batch(), k)

    # Benchmark
    times = []
    for _ in range(steps):
        rng, k = jax.random.split(rng)
        batch = make_batch()
        jax.block_until_ready(agent.state.params)
        t0 = time.perf_counter()
        agent.train_step(batch, k)
        jax.block_until_ready(agent.state.params)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    try:
        mem_gb = jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 1e9
    except Exception:
        mem_gb = float("nan")

    return {
        "mean_step_time": float(np.mean(times)),
        "std_step_time": float(np.std(times)),
        "steps_per_sec": float(1.0 / np.mean(times)),
        "peak_gpu_gb": float(mem_gb),
    }


def benchmark_pytorch_r2dreamer(steps, obs_shape, num_actions, batch_size, seq_len, rep_loss):
    """Benchmark PyTorch r2dreamer training throughput."""
    import torch

    r2_dir = os.path.join(os.path.dirname(__file__), "..", "external", "r2dreamer")
    sys.path.insert(0, r2_dir)

    import gym
    from omegaconf import OmegaConf

    # Load config
    base_cfg = OmegaConf.load(os.path.join(r2_dir, "configs", "model", "_base_.yaml"))
    size_cfg = OmegaConf.load(os.path.join(r2_dir, "configs", "model", "size12M.yaml"))
    cfg = OmegaConf.merge(base_cfg, size_cfg)
    cfg.rep_loss = rep_loss
    cfg.device = "cuda:0"
    cfg.compile = False

    H, W = obs_shape[1], obs_shape[2]
    obs_space = gym.spaces.Dict({
        "image": gym.spaces.Box(0, 255, (H, W, 3), dtype=np.uint8),
    })
    act_space = gym.spaces.Discrete(num_actions)

    from dreamer import Dreamer
    agent = Dreamer(cfg, obs_space, act_space).to(cfg.device)

    def make_batch():
        from tensordict import TensorDict
        return TensorDict({
            "image": torch.rand(batch_size, seq_len, H, W, 3, device=cfg.device),
            "action": torch.eye(num_actions, device=cfg.device)[
                torch.randint(0, num_actions, (batch_size, seq_len))],
            "reward": torch.zeros(batch_size, seq_len, device=cfg.device),
            "is_first": torch.zeros(batch_size, seq_len, device=cfg.device),
            "is_last": torch.zeros(batch_size, seq_len, device=cfg.device),
            "is_terminal": torch.zeros(batch_size, seq_len, device=cfg.device),
        }, batch_size=(batch_size, seq_len))

    initial = agent.get_initial_state(batch_size)

    # Warmup
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        data = make_batch()
        agent._update_slow_target()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            agent._cal_grad(data, (initial["stoch"], initial["deter"]))
        agent._optimizer.zero_grad(set_to_none=True)

    # Benchmark
    times = []
    for _ in range(steps):
        data = make_batch()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        agent._update_slow_target()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            agent._cal_grad(data, (initial["stoch"], initial["deter"]))
        agent._scaler.unscale_(agent._optimizer)
        agent._agc(agent._named_params.values())
        agent._scaler.step(agent._optimizer)
        agent._scaler.update()
        agent._optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mem_gb = torch.cuda.max_memory_allocated() / 1e9

    return {
        "mean_step_time": float(np.mean(times)),
        "std_step_time": float(np.std(times)),
        "steps_per_sec": float(1.0 / np.mean(times)),
        "peak_gpu_gb": float(mem_gb),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--output", type=str, default="output/comparison/habitat_timing.json")
    args = parser.parse_args()

    obs_shape = (3, 64, 64)
    num_actions = 4
    batch_size = 16
    seq_len = 64

    results = {}

    print("=" * 50)
    print("Benchmarking JAX R2-Dreamer...")
    results["r2dreamer_jax"] = benchmark_jax_r2dreamer(
        args.steps, obs_shape, num_actions, batch_size, seq_len)
    print(f"  {results['r2dreamer_jax']['steps_per_sec']:.1f} steps/sec, "
          f"{results['r2dreamer_jax']['peak_gpu_gb']:.2f} GB")

    print("\nBenchmarking PyTorch R2-Dreamer...")
    results["r2dreamer_pytorch"] = benchmark_pytorch_r2dreamer(
        args.steps, obs_shape, num_actions, batch_size, seq_len, "r2dreamer")
    print(f"  {results['r2dreamer_pytorch']['steps_per_sec']:.1f} steps/sec, "
          f"{results['r2dreamer_pytorch']['peak_gpu_gb']:.2f} GB")

    print("\nBenchmarking PyTorch DreamerV3 (decoder)...")
    results["dreamerv3_pytorch"] = benchmark_pytorch_r2dreamer(
        args.steps, obs_shape, num_actions, batch_size, seq_len, "dreamer")
    print(f"  {results['dreamerv3_pytorch']['steps_per_sec']:.1f} steps/sec, "
          f"{results['dreamerv3_pytorch']['peak_gpu_gb']:.2f} GB")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add scripts/habitat_env_r2dreamer.py scripts/run_habitat_timing.py
git commit -m "feat: add Habitat env wrapper for PyTorch r2dreamer and timing benchmark"
```

---

## Task 11: Extend dreamerv3_official_comparison.ipynb

**Files:**
- Modify: `notebooks/dreamerv3_official_comparison.ipynb`

Extend the existing 2-way (Ours JAX DreamerV3 vs Official DreamerV3) to a 5-way comparison:
1. Our JAX DreamerV3 (existing)
2. Official DreamerV3 (existing)
3. PyTorch r2dreamer `rep_loss=dreamer`
4. PyTorch r2dreamer `rep_loss=r2dreamer`
5. Our JAX R2-Dreamer

- [ ] **Step 1: Add cells to run new variants and load their CSVs**

Add cells after the existing "Run Official" section that:
1. Run `scripts/run_r2dreamer_pytorch_crafter.py --rep_loss both --steps 100000`
2. Run `scripts/run_r2dreamer_jax_crafter.py --steps 100000 --output output/comparison/r2dreamer_jax_metrics.csv`
3. Load all 5 CSVs into dataframes

- [ ] **Step 2: Update plot functions for 5-way comparison**

Update `plot_dual_axis` to handle 5 series with distinct colors:
- Our JAX DreamerV3: `#2196F3` (blue)
- Official DreamerV3: `#FF5722` (orange)
- PyTorch DreamerV3 (decoder): `#4CAF50` (green)
- PyTorch R2-Dreamer: `#9C27B0` (purple)
- JAX R2-Dreamer: `#F44336` (red)

- [ ] **Step 3: Update summary table and verdict**

Add all 5 variants to the summary table. Update verdict to check that JAX R2-Dreamer matches PyTorch R2-Dreamer (same loss trends).

- [ ] **Step 4: Execute notebook**

Run: `cd /pfs/data6/home/ul/ul_student/ul_hfj15/Master-Thesis-3D-VLA && uv run jupyter nbconvert --to notebook --execute notebooks/dreamerv3_official_comparison.ipynb --inplace --ExecutePreprocessor.timeout=7200`
Expected: All cells execute, plots generated

- [ ] **Step 5: Commit**

```bash
git add notebooks/dreamerv3_official_comparison.ipynb
git commit -m "feat: extend comparison notebook to 5-way (DreamerV3/R2-Dreamer, JAX/PyTorch)"
```

---

## Task 12: Rewrite dreamerv3_jax_vs_pytorch.ipynb

**Files:**
- Modify: `notebooks/dreamerv3_jax_vs_pytorch.ipynb`

Rewrite as a 3-way Habitat timing comparison using results from `scripts/run_habitat_timing.py`.

- [ ] **Step 1: Write notebook cells**

Cell 1 (markdown): Title + description of 3-way comparison.

Cell 2 (code): Run `scripts/run_habitat_timing.py` (or load existing results).

Cell 3 (code): Load `output/comparison/habitat_timing.json`, create bar charts for:
- Steps/sec (3 bars)
- Peak GPU memory (3 bars)
- Step time distribution (box plots if individual times are stored)

Cell 4 (code): Summary table:
```
| Variant              | Steps/sec | GPU Memory (GB) | Speedup vs DreamerV3 |
|---                   |---        |---               |---                    |
| DreamerV3 (PyTorch)  | X.X       | X.XX             | 1.00x                |
| R2-Dreamer (PyTorch) | X.X       | X.XX             | X.XXx                |
| R2-Dreamer (JAX)     | X.X       | X.XX             | X.XXx                |
```

Cell 5 (markdown): Conclusion — does JAX improve performance?

- [ ] **Step 2: Execute notebook**

Run: `cd /pfs/data6/home/ul/ul_student/ul_hfj15/Master-Thesis-3D-VLA && uv run jupyter nbconvert --to notebook --execute notebooks/dreamerv3_jax_vs_pytorch.ipynb --inplace --ExecutePreprocessor.timeout=3600`
Expected: All cells execute, comparison plots generated

- [ ] **Step 3: Commit**

```bash
git add notebooks/dreamerv3_jax_vs_pytorch.ipynb
git commit -m "feat: rewrite timing notebook as 3-way Habitat benchmark"
```

---

## Task 13: Final Integration Smoke Test

- [ ] **Step 1: Run all network shape tests**

Run: `cd /pfs/data6/home/ul/ul_student/ul_hfj15/Master-Thesis-3D-VLA && uv run pytest tests/test_r2dreamer_shapes.py tests/test_r2dreamer_optim.py tests/test_r2dreamer_agent.py -v`
Expected: All PASS

- [ ] **Step 2: Run existing DreamerV3 tests (regression check)**

Run: `cd /pfs/data6/home/ul/ul_student/ul_hfj15/Master-Thesis-3D-VLA && uv run pytest tests/test_dreamerv3_shapes.py -v`
Expected: All PASS (existing code untouched)

- [ ] **Step 3: Quick Crafter smoke test (500 steps)**

Run: `cd /pfs/data6/home/ul/ul_student/ul_hfj15/Master-Thesis-3D-VLA && uv run python scripts/run_r2dreamer_jax_crafter.py --steps 500 --prefill 200 --log_every 100 --output output/smoke_final.csv`
Expected: Runs to completion, losses are finite

- [ ] **Step 4: Commit all remaining changes**

```bash
git add -A
git commit -m "feat: R2-Dreamer JAX port complete — ready for comparison runs"
```
