# VGGT → R2Dreamer Call Chain

End-to-end data flow for one environment step when `encoder_type="vggt"`.

## Chain (top → bottom)

```
HabitatObjectNavEnv.step()
   └─ obs_dict = {"image": uint8 (3,518,518), "is_first": bool}
        │
        ▼
VGGTObsAdapter.transform(obs_dict)               [scripts/run_jax_habitat_vggt.py:44]
   ├─ VGGTFeatureExtractor.extract(rgb)          [vggt/feature_extractor.py:83]   ← PyTorch, no_grad, KV-cached
   │     └─ {"world_points":(37,37,3) f32, "camera_pose":(9,) f32}
   ├─ _flatten_vggt(out) → np.float32 (4116,)    [scripts/run_jax_habitat_vggt.py:25]   ← 4107 + 9
   └─ returns (features, agent_obs={"features":..., "is_first":...})
        │
        ▼
R2DreamerAgent.act(agent_obs)                    [r2dreamer/agent.py:178]
   ├─ obs = jnp.array(obs_dict["features"][None])           ← PyTorch→JAX boundary (via numpy)
   └─ self._jit_act(...)                         [r2dreamer/agent.py:222]
        ├─ embed = VGGTEncoder.apply(obs)        [r2dreamer/networks.py:422]   ← single Dense 4116→1024
        ├─ new_stoch, new_deter = RSSM.apply(...)
        └─ logits = actor.apply(feat) → action_int
```

## Architectural notes

- **Framework boundary** is the numpy `features` array. PyTorch tensors stay in `VGGTFeatureExtractor`; JAX tensors start at `agent.act`. No shared autograd graph.
- **VGGTEncoder is a single Dense layer** (4116 → 1024). The "learning" between 3D features and the world model is one linear projection — debugging effort belongs *upstream* (feature quality) and *downstream* (RSSM utilization), not in the encoder itself.
- **Encoder switch is config-only** (`encoder_type: "cnn" | "vggt"`). All branching is at `_make_encoder` (agent.py:50) and `act()` (agent.py:191). A/B comparisons need only flip the config.
- **VGGT model is frozen** (`p.requires_grad = False`) and runs under `torch.no_grad() + amp.autocast`. KV-cache (`_past_key_values`, `_past_key_values_camera`) carries cross-frame context; `extractor.reset()` is called at every episode boundary via `obs_adapter.on_episode_reset`.

## Prefill vs train-loop fork (trainer.py)

| Phase | Calls `obs_adapter.transform`? | Calls `agent.act`? |
|-------|--------------------------------|--------------------|
| `_prefill` (trainer.py:313) | YES — every step | NO — uses `np.random.randint` |
| `_train_loop` (trainer.py:344) | YES | YES (trainer.py:365) |

Implication for debugging: BPs #1–#2 fire during prefill; BPs #3–#5 only fire post-prefill. Use `--prefill 0` (or a small number) to reach `agent.act` on step 1.

## High-leverage breakpoints

| # | File:Line | Variable to inspect | Why |
|---|-----------|---------------------|-----|
| 1 | `vggt/feature_extractor.py:171` | `world_points_np` shape/range, `camera_pose_np` | Confirms 3D backbone output. NaN/zero here = PyTorch model is the bug. |
| 2 | `scripts/run_jax_habitat_vggt.py:45` | `features.shape == (4116,)`, dtype, NaN check | Shape contract (4107+9). Most common silent failure when patch grid / resolution changes. |
| 3 | `r2dreamer/agent.py:192` | `obs.shape == (1, 4116)`, dtype, `is_first` | PyTorch→JAX handoff. Best **single starting BP** — halves the search space immediately. |
| 4 | `r2dreamer/networks.py:425` | encoder return `.shape == (1, 1024)` | Inside JIT — use `jax.debug.print`, not `breakpoint()`. |
| 5 | `r2dreamer/agent.py:226` | `embed.shape`, `new_stoch.shape`, `new_deter.shape` | Inside JIT. Where world-model state actually updates from 3D embedding. **Thesis-critical**: compare CNN vs VGGT trajectories of `new_stoch`/`new_deter`. |

## JIT caveat

BPs #4 and #5 sit inside `self._jit_act = jax.jit(self._act_jit)` (agent.py:172). Python `breakpoint()` traces with abstract values, not real arrays. To inspect:
- Temporarily replace `self._jit_act = jax.jit(self._act_jit)` with `self._jit_act = self._act_jit`, OR
- Use `jax.debug.print("embed shape: {}", embed.shape)` which works inside JIT.

## Findings from instrumented run (2026-04-25, `--prefill 0 --steps 1`)

Log: `output/debug_session/bp1/run.log`. All values from BP prints injected at the 5 locations above.

### Healthy signals
- **VGGT geometry is sane**: `world_points` shape `(37,37,3) f32`, no NaN/Inf, Z strictly positive (in front of camera), per-axis ranges plausible (sub-meter to ~1.6m). Camera_pose ~10⁻³ near identity for first frame after reset.
- **Streaming KV-cache works**: `frame_idx` resets to 1 on `env.reset()` (cleared by `obs_adapter.on_episode_reset → extractor.reset()`), increments to 2 on within-episode `env.step()`. StreamVGGT context carries forward as designed.
- **Shape contract intact end-to-end**: `(4116,) f32` at BP#2 → `(1,4116) f32` at BP#3 → `(1,1024)` at BP#4. No silent dtype upcasts at the PyTorch→JAX boundary.
- **Encoder distribution healthy at init**: `embed.std()=0.52`, range `[-1.6, +1.5]`, mean ≈ 0. Matches Lecun-init prediction for `Dense(4116→1024)` over input variance ~0.5.

### Surprises / red flags

#### 1. `camera_pose` is *episode-relative*, not scene-absolute
StreamVGGT defines pose w.r.t. the first frame after KV-cache reset. So:
- After `env.reset()`: pose ≈ identity (defines new world frame)
- Mid-episode: pose drifts proportional to ego-motion since reset
- Across episodes: there is no global localization signal

For ObjectNav this is fine (the task wants generalization, not memorization), but worth knowing for anyone interpreting the encoder behavior.

#### 2. `new_deter == 0` at the very first acting step (BP#5), and grows ridiculously slowly thereafter
RSSM's GRU computes zero output when given all-zero state + zero inputs (provable from zero-bias init). On step 0 of an episode, `prev_stoch`, `prev_deter`, `prev_action` are all zeroed by `is_first` handling at agent.py:198–201. The `embed` argument is passed but only feeds the *stoch* path during act-time — `new_deter` only \"catches up\" to observation at step 1+.

**Verified by `--steps 5` re-run** (`output/debug_session/bp1/run5.log`):

| Step | `embed.std` | `new_deter.std` | growth |
|------|-------------|-----------------|--------|
| 0    | 0.517       | **0.0000**      | — |
| 1    | 0.520       | 0.00271         | first activation |
| 2    | 0.533       | 0.00373         | +37% |
| 3    | 0.565       | 0.00433         | +16% |
| 4    | 0.527       | 0.00527         | +22% |

`new_stoch.std` stays at exactly 0.2421 (uniform-categorical max) the whole time — posterior is maximally uncertain at init.

**Two findings**:
1. **1-step embed→deter lag confirmed** (deter is exactly 0 on step 0, nonzero from step 1).
2. **Slow growth bonus finding**: even after 5 steps, `new_deter.std ≈ 0.005` — about **50× smaller than `new_stoch.std`** and **100× smaller than `embed.std`**. The actor reads `feat = [stoch, deter]` and at this stage effectively only sees stoch.

**Implication for thesis**: pose / 3D ego-motion information that needs to integrate over time (loop detection, trajectory reasoning) is invisible to the actor for many steps until deter accumulates magnitude. Combined with #87 (pose dilution at input), this creates a compound delay before 3D-specific information can influence policy. Tracked in [#90](https://github.com/LSailer/Master-Thesis-3D-VLA/issues/90).

#### 3. `new_stoch.mean() = exactly 1/16 = 0.0625`
Each of 32 categoricals chose a one-hot bin; bins are uniformly distributed across 16 options → maximum-entropy posterior. Healthy for step 0 (no history yet); should *decrease* over training as RSSM learns confident posteriors. Worth logging as a metric.

### Architectural concerns surfaced (parked as GH issues)

| Issue | Layer | Concern |
|-------|-------|---------|
| [#87](https://github.com/LSailer/Master-Thesis-3D-VLA/issues/87) | Input-side | `camera_pose` is 0.22% of dims AND ~100× smaller magnitude than `world_points` → effective signal contribution ~0.002% at init. Risks burying the ego-motion signal. |
| [#88](https://github.com/LSailer/Master-Thesis-3D-VLA/issues/88) | Framework boundary | `extract()` does `.cpu().numpy()`, then `act()` does `jnp.array()` — GPU→CPU→GPU round-trip on every step. Bandwidth trivial but CUDA sync costs latency. Profile first, fix with DLPack if warranted. |
| [#89](https://github.com/LSailer/Master-Thesis-3D-VLA/issues/89) | Encoder architecture | `VGGTEncoder` is a single linear `Dense(4116→1024)` — cannot model multiplicative `pose × geometry` interactions. Orthogonal to #87. |

All three are confounders for the 3D-vs-2D thesis claim: if 3D ties 2D, we cannot distinguish \"3D doesn't help\" from \"our encoder is sabotaging it.\"

### Metrics worth logging during training (recommendations from this walkthrough)
- `embed.std()` per step — alarm if drifts to 0 (dead) or >5 (saturating)
- `new_stoch.std() / max_std` — should decrease over training (posterior gets confident)
- `new_deter.std()` — should be nonzero by step 1 of every episode

### JIT caveats encountered
- `jax.debug.print` correctly defers to runtime (printed once per real call, not during tracing)
- Inside JIT, `out.shape` returns traced int arrays (`Array(1024, dtype=int32)`) instead of plain Python ints — cosmetic only
- `R2DreamerAgent.__init__` triggers two encoder calls with dummy zeros (agent.py:84–85) to discover `embed_size` — these show as `mean=std=0` in BP#4 logs and are not real forward passes
