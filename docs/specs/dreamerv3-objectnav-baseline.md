# DreamerV3 ObjectNav Baseline Training Spec

## 1. Overview

Train DreamerV3 from pixels on HM3D ObjectNav. Measure success rate + SPL.
Baseline for thesis — later compare with UNITE 3D feature injection.

## 2. Environment

**Primary: HM3D ObjectNav (`src/dreamerv3/env_habitat.py`)**
- 4 discrete actions: STOP, FORWARD, LEFT, RIGHT
- RGB 256x256, 500-step episodes
- HM3D v2: 800 train scenes, 200 val scenes, 6 object categories

**Goal conditioning:** agent receives RGB-only — `env_habitat.py` does NOT pass `objectgoal` category to the agent. This baseline measures general exploration + navigation learning, not goal-directed search. Adding goal conditioning (e.g. one-hot `objectgoal_sensor` concatenated to encoder) is a known gap for future work.

**Sensor gap:** RGB-only — no depth, no GPS/compass. Results not directly comparable to methods using privileged sensors (e.g. DD-PPO with depth+GPS).

## 3. World Model — handling different scenes

WM sees only RGB, no scene ID. Must generalize across scenes.

- RSSM learns scene-agnostic dynamics (walls block movement, turns rotate view)
- Replay buffer mixes episodes from all scenes → WM learns scene distribution
- Episode reset (`is_first`) zeros RSSM state → no cross-scene leakage
- Categorical latent (32x32) = multimodal capacity for diverse layouts
- ObjectNav dynamics largely scene-invariant → encoder forced to learn generalizable features
- DreamerV3 paper: single config across 150+ tasks

**Monitor:** WM loss convergence, reconstruction quality on unseen scenes, replay diversity

## 4. Reward

**Default: geodesic_delta (`env_habitat.py:74-83`)**
- `reward = prev_geodesic_dist - curr_geodesic_dist + 10.0 * success`
- Dense, path-aware, symlog-transformed in WM

Reward ablation (geodesic_delta vs sparse vs euclidean vs slack penalty) tracked separately — see GitHub issue.

## 5. Training Pipeline

**Existing flow (`src/dreamerv3/train.py`, no major changes):**
1. Prefill 5k random steps → replay buffer
2. act → env.step → buffer.add → sample batch → train WM + actor-critic
3. Batch: 16 seq x 64 timesteps
4. Imagination: 15-step rollouts
5. Checkpoint every 50k steps, log to W&B

**Compute plan (48h single H100):**
- 5M env steps target (≥10k episodes, depends on early termination rate)
- **Step 0: profiling run** — 10k steps to measure actual steps/min, then confirm 5M fits in 48h
- If 5M steps > 48h after profiling → fallback: (a) reduce to 2.5M, (b) multi-env vectorization, (c) request 2x GPU allocation
- 3 seeds minimum for any reported result. 1 seed for initial validation only.

**Metrics:**
- Primary: `success_rate`, `spl`
- Episode: `episode_reward`, `episode_length`
- WM: `recon_loss`, `kl_loss`, `reward_loss`, `cont_loss`
- Behavior: `actor_loss`, `critic_loss`, `entropy`, `imag_return`

## 6. Evaluation Protocol

- Eval on `val` split every 100k steps (no training)
- **Both** greedy (argmax) and stochastic policy logged; greedy = primary metric
- Report: success rate, SPL, distance-to-goal at termination
- **Not yet implemented:** greedy eval loop, `eval_every`/`eval_episodes` flags, val-split eval, metric logging. Gate 4 prereq: implement eval loop (argmax policy mode, val-split eval, metric logging).
- **Success target:** unknown — no prior DreamerV3-on-ObjectNav-RGB baseline exists. Profiling run (Gate 3) establishes first data point.
- **Sensor gap note:** agent is RGB-only (no depth, no GPS/compass). Performance not comparable to methods using privileged sensors.

## 7. Implementation via TDD Cycles

Strict TDD: write one failing test → implement → verify green → commit → next cycle.
Each cycle specifies: feature, test, implementation hint, pass command, commit message.

Tests go in existing files: `test_dreamerv3_shapes.py` (unit, no Habitat/GPU), `test_dreamerv3_integration.py` (needs Habitat+GPU).

### Cycle 1: `__main__.py` entry point

**Feature:** `python -m src.dreamerv3` invokes `train.main()`

**Test** (`tests/test_dreamerv3_shapes.py`):
```python
class TestEntryPoint:
    def test_main_module_exists(self):
        """__main__.py exists and imports main."""
        import importlib
        mod = importlib.import_module("src.dreamerv3.__main__")
        assert hasattr(mod, "main") or callable(getattr(mod, "main", None)) is False
        # Verify it re-exports train.main
        from src.dreamerv3.train import main
        import inspect
        source = inspect.getsource(mod)
        assert "train" in source and "main" in source
```

**Implementation:** Create `src/dreamerv3/__main__.py`:
```python
from .train import main
if __name__ == "__main__":
    main()
```

**Pass:** `uv run pytest tests/test_dreamerv3_shapes.py::TestEntryPoint -v`

**Commit:** `add dreamerv3 __main__.py entry point`

---

### Cycle 2: Greedy action mode

**Feature:** `agent.act(obs, key, training=False)` uses `jnp.argmax` instead of sampling

**Test** (`tests/test_dreamerv3_shapes.py`):
```python
class TestGreedyAction:
    def test_greedy_is_deterministic(self, cfg, rng):
        """training=False → argmax → same action for same state."""
        from src.dreamerv3.agent import DreamerAgent
        agent = DreamerAgent(cfg, rng)
        obs = {"image": jnp.zeros(cfg.obs_shape, dtype=jnp.uint8), "is_first": True}
        k1, k2 = jax.random.split(rng)
        a1 = agent.act(obs, k1, training=False)
        a2 = agent.act(obs, k2, training=False)
        assert a1 == a2, "greedy mode must be deterministic regardless of rng key"

    def test_stochastic_uses_sampling(self, cfg, rng):
        """training=True → categorical sampling (default behavior)."""
        import inspect
        from src.dreamerv3.agent import DreamerAgent
        source = inspect.getsource(DreamerAgent._act_forward)
        assert "training" in inspect.signature(DreamerAgent._act_forward).parameters or \
               "argmax" in source or "categorical" in source
```

**Implementation** (`src/dreamerv3/agent.py`):
- Pass `training` through `act()` → `_act_jit` → `_act_forward`
- In `_act_forward`: `action = jnp.where(training, jax.random.categorical(rng_key, logits), jnp.argmax(logits, axis=-1))`
- Need two JIT'd versions or use `jax.lax.cond` / `jnp.where` (static arg or functional branch)
- Simplest: use `functools.partial` with `static_argnums` for `training` flag

**Pass:** `uv run pytest tests/test_dreamerv3_shapes.py::TestGreedyAction -v`

**Commit:** `add greedy action mode (training=False → argmax)`

---

### Cycle 3: `eval_every` / `eval_episodes` config fields

**Feature:** `DreamerConfig` has `eval_every` and `eval_episodes` fields

**Test** (`tests/test_dreamerv3_shapes.py`):
```python
class TestEvalConfig:
    def test_eval_fields_exist(self, cfg):
        assert hasattr(cfg, "eval_every")
        assert hasattr(cfg, "eval_episodes")
        assert isinstance(cfg.eval_every, int)
        assert isinstance(cfg.eval_episodes, int)

    def test_eval_defaults(self, cfg):
        assert cfg.eval_every == 100_000
        assert cfg.eval_episodes == 10
```

**Implementation** (`src/dreamerv3/configs.py`):
Add to `DreamerConfig`:
```python
# Eval
eval_every: int = 100_000
eval_episodes: int = 10
```

**Pass:** `uv run pytest tests/test_dreamerv3_shapes.py::TestEvalConfig -v`

**Commit:** `add eval_every/eval_episodes config fields`

---

### Cycle 4: `imag_return` metric regression guard

**Feature:** `train_step` returns `imag_return` in metrics dict (already exists — this is a regression guard)

**Test** (`tests/test_dreamerv3_shapes.py`):
```python
class TestImaginationMetrics:
    def test_imag_return_in_metrics(self, cfg, rng):
        """train_step must return imag_return metric."""
        from src.dreamerv3.agent import DreamerAgent
        agent = DreamerAgent(cfg, rng)
        B, T = cfg.batch_size, cfg.seq_len
        batch = {
            "obs": jnp.zeros((B, T, *cfg.obs_shape)),
            "actions": jnp.zeros((B, T), dtype=jnp.int32),
            "rewards": jnp.zeros((B, T)),
            "dones": jnp.zeros((B, T)),
            "is_first": jnp.zeros((B, T)),
        }
        metrics = agent.train_step(batch, rng)
        assert "imag_return" in metrics, "imag_return must be in train_step metrics"
        assert isinstance(metrics["imag_return"], float)
```

**Implementation:** None expected — should pass immediately. If not, check `_train_behavior` returns `imag_return` key.

**Pass:** `uv run pytest tests/test_dreamerv3_shapes.py::TestImaginationMetrics -v`

**Commit:** `add imag_return regression guard test`

---

### Cycle 5: Checkpoint save/load roundtrip

**Feature:** `agent.save()` then `agent.load()` preserves params (already exists — regression guard)

**Test** (`tests/test_dreamerv3_shapes.py`):
```python
class TestCheckpoint:
    def test_save_load_roundtrip(self, cfg, rng, tmp_path):
        """Save → load → params match."""
        from src.dreamerv3.agent import DreamerAgent
        agent = DreamerAgent(cfg, rng)
        original_params = jax.tree.map(lambda x: x.copy(), agent.wm_state.params)
        agent.save(str(tmp_path))
        # Clobber params
        agent.wm_state = agent.wm_state.replace(
            params=jax.tree.map(jnp.zeros_like, agent.wm_state.params)
        )
        agent.load(str(tmp_path))
        for orig, loaded in zip(
            jax.tree.leaves(original_params),
            jax.tree.leaves(agent.wm_state.params),
        ):
            assert jnp.allclose(orig, loaded, atol=1e-6)

    def test_checkpoint_file_exists(self, cfg, rng, tmp_path):
        from src.dreamerv3.agent import DreamerAgent
        agent = DreamerAgent(cfg, rng)
        agent.save(str(tmp_path))
        assert (tmp_path / "checkpoint.pkl").exists()
```

**Implementation:** None expected — should pass immediately.

**Pass:** `uv run pytest tests/test_dreamerv3_shapes.py::TestCheckpoint -v`

**Commit:** `add checkpoint roundtrip regression guard test`

---

### Cycle 6: Checkpoint resume in `train.py`  *(depends on Cycle 5)*

**Feature:** `train.py` loads existing checkpoint from `logdir` before training if one exists

**Test** (`tests/test_dreamerv3_shapes.py`):
```python
class TestCheckpointResume:
    def test_train_loads_existing_checkpoint(self):
        """train.main() must call agent.load() when checkpoint exists."""
        import inspect
        from src.dreamerv3 import train
        source = inspect.getsource(train.main)
        # Must check for checkpoint and call load
        assert "load" in source, "train.main must call agent.load for resume"
        assert "checkpoint" in source, "train.main must check for existing checkpoint"
```

**Implementation** (`src/dreamerv3/train.py`):
After agent creation, before prefill:
```python
ckpt_path = os.path.join(config.logdir, "checkpoint.pkl")
if os.path.exists(ckpt_path):
    agent.load(config.logdir)
    print(f"Resumed from {ckpt_path}")
```

**Pass:** `uv run pytest tests/test_dreamerv3_shapes.py::TestCheckpointResume -v`

**Commit:** `add checkpoint resume in train.py`

---

### Cycle 7: `run_eval()` function  *(depends on Cycle 2)*

**Feature:** standalone `run_eval(agent, env, num_episodes, greedy)` → returns `{success_rate, spl, mean_reward}`

**Test** (`tests/test_dreamerv3_shapes.py`):
```python
class TestRunEval:
    def test_run_eval_exists(self):
        """run_eval function exists in train module."""
        from src.dreamerv3.train import run_eval
        import inspect
        sig = inspect.signature(run_eval)
        params = list(sig.parameters.keys())
        assert "agent" in params
        assert "env" in params

    def test_run_eval_signature_has_greedy(self):
        """run_eval accepts greedy flag."""
        from src.dreamerv3.train import run_eval
        import inspect
        sig = inspect.signature(run_eval)
        assert "greedy" in sig.parameters or "training" in sig.parameters

    def test_run_eval_returns_metrics_dict(self):
        """run_eval returns dict with success_rate, spl, mean_reward keys."""
        from src.dreamerv3.train import run_eval
        import inspect
        source = inspect.getsource(run_eval)
        assert "success_rate" in source
        assert "spl" in source
```

**Implementation** (`src/dreamerv3/train.py`):
```python
def run_eval(agent, env, num_episodes, rng_key, greedy=True):
    """Run eval episodes, return {success_rate, spl, mean_reward}."""
    successes, spls, rewards = [], [], []
    for _ in range(num_episodes):
        obs = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            rng_key, k = jax.random.split(rng_key)
            action = agent.act(obs, k, training=not greedy)
            obs = env.step(action)
            ep_reward += obs["reward"]
            done = obs["done"]
        successes.append(obs.get("success", 0.0))
        spls.append(obs.get("spl", 0.0))
        rewards.append(ep_reward)
    return {
        "success_rate": float(np.mean(successes)),
        "spl": float(np.mean(spls)),
        "mean_reward": float(np.mean(rewards)),
    }
```

**Pass:** `uv run pytest tests/test_dreamerv3_shapes.py::TestRunEval -v`

**Commit:** `add run_eval() function`

---

### Cycle 8: Eval loop in training loop  *(depends on Cycles 3, 7)*

**Feature:** `train.py` main loop calls `run_eval` every `eval_every` steps on a val-split env

**Test** (`tests/test_dreamerv3_shapes.py`):
```python
class TestEvalInTrainLoop:
    def test_train_main_calls_run_eval(self):
        """train.main must call run_eval at eval_every intervals."""
        import inspect
        from src.dreamerv3 import train
        source = inspect.getsource(train.main)
        assert "run_eval" in source, "train.main must call run_eval"
        assert "eval_every" in source, "train.main must use eval_every"

    def test_train_creates_val_env(self):
        """train.main must create a separate val-split env for eval."""
        import inspect
        from src.dreamerv3 import train
        source = inspect.getsource(train.main)
        assert "val" in source, "train.main must create val-split env"

    def test_train_logs_eval_metrics(self):
        """train.main must log eval/ prefixed metrics."""
        import inspect
        from src.dreamerv3 import train
        source = inspect.getsource(train.main)
        assert "eval/" in source, "train.main must log eval/-prefixed metrics"
```

**Implementation** (`src/dreamerv3/train.py`):
- Create `val_env = HabitatObjectNavEnv(config._replace(split="val_mini"))` at startup (use `dataclasses.replace`)
- In main loop, after training step:
```python
if step > 0 and step % config.eval_every == 0:
    rng_key, eval_key = jax.random.split(rng_key)
    greedy_metrics = run_eval(agent, val_env, config.eval_episodes, eval_key, greedy=True)
    rng_key, eval_key = jax.random.split(rng_key)
    stoch_metrics = run_eval(agent, val_env, config.eval_episodes, eval_key, greedy=False)
    eval_log = {f"eval/{k}_greedy": v for k, v in greedy_metrics.items()}
    eval_log.update({f"eval/{k}_stochastic": v for k, v in stoch_metrics.items()})
    print(f"[step {step}] eval: {eval_log}")
    if use_wandb:
        wandb.log(eval_log, step=step)
```
- Close `val_env` at end

**Pass:** `uv run pytest tests/test_dreamerv3_shapes.py::TestEvalInTrainLoop -v`

**Commit:** `add eval loop to training (val-split, greedy+stochastic)`

---

### Cycle 9: Stochastic eval metrics  *(depends on Cycle 7)*

**Feature:** eval logs both greedy and stochastic metrics with distinct keys

**Test** (`tests/test_dreamerv3_shapes.py`):
```python
class TestStochasticEvalMetrics:
    def test_eval_loop_logs_both_modes(self):
        """Training loop must log both greedy and stochastic eval metrics."""
        import inspect
        from src.dreamerv3 import train
        source = inspect.getsource(train.main)
        assert "greedy" in source and "stochastic" in source, \
            "must log both greedy and stochastic eval"
        assert "eval/success_rate_greedy" in source or \
               ("eval/" in source and "greedy" in source), \
            "must have eval/..._greedy metric keys"
```

**Implementation:** Already covered by Cycle 8 if both modes are wired. Verify keys match `eval/success_rate_greedy`, `eval/spl_greedy`, `eval/success_rate_stochastic`, `eval/spl_stochastic`.

**Pass:** `uv run pytest tests/test_dreamerv3_shapes.py::TestStochasticEvalMetrics -v`

**Commit:** `add stochastic eval metrics test`

---

### Cycle 10: End-to-end eval on Habitat  *(depends on Cycles 2, 7, 9; needs Habitat+GPU)*

**Feature:** Full integration — eval runs on real Habitat val env, returns valid metrics

**Test** (`tests/test_dreamerv3_integration.py`):
```python
def test_eval_loop_on_habitat(small_config):
    """Run eval on real Habitat env — greedy + stochastic produce valid metrics."""
    import jax
    from src.dreamerv3.agent import DreamerAgent
    from src.dreamerv3.env_habitat import HabitatObjectNavEnv
    from src.dreamerv3.train import run_eval

    eval_config = dataclasses.replace(small_config, split="val_mini", max_episode_steps=20)
    env = HabitatObjectNavEnv(eval_config)
    rng = jax.random.PRNGKey(0)
    agent = DreamerAgent(eval_config, rng)
    try:
        rng, k = jax.random.split(rng)
        greedy = run_eval(agent, env, num_episodes=2, rng_key=k, greedy=True)
        rng, k = jax.random.split(rng)
        stoch = run_eval(agent, env, num_episodes=2, rng_key=k, greedy=False)

        for metrics in [greedy, stoch]:
            assert "success_rate" in metrics
            assert "spl" in metrics
            assert "mean_reward" in metrics
            assert 0.0 <= metrics["success_rate"] <= 1.0
            assert 0.0 <= metrics["spl"] <= 1.0
    finally:
        env.close()
```

**Implementation:** None — all pieces assembled from Cycles 2, 7-9. This is the integration capstone.

**Pass:** `uv run pytest tests/test_dreamerv3_integration.py::test_eval_loop_on_habitat -v`

**Commit:** `add end-to-end eval integration test`

---

### Post-TDD Gates

After all 10 cycles pass, run these gates in order:

**Static checks (no GPU):**
```bash
uv run pytest tests/test_dreamerv3_shapes.py -v
uv run mypy src/dreamerv3/ --ignore-missing-imports
uv run ruff check src/dreamerv3/
uv run ruff format --check src/dreamerv3/
```

**Smoke test (needs Habitat + GPU):**
```bash
uv run python -m src.dreamerv3 \
  --total_steps 100 --prefill_steps 50 --split val_mini \
  --logdir /tmp/dreamer_smoke_$$
ls /tmp/dreamer_smoke_$$/checkpoint.pkl  # verify checkpoint exists
```

**Profiling run (10k steps, ~30 min):**
```bash
uv run python -m src.dreamerv3 \
  --total_steps 10000 --prefill_steps 1000 \
  --logdir output/dreamerv3/profile-run
```
Pass: throughput logged, `recon_loss` decreasing, no NaN, 5M steps fits in 48h.

**Full training (human triggers):**
```bash
sbatch scripts/slurm/train_dreamer.sbatch
```
Post-training: `ls output/dreamerv3/run-*/checkpoint.pkl`, W&B has >4M steps, `success_rate > 0`.

### Cycle dependency graph

```
1 (__main__)  ──────────────────────────────────────┐
2 (greedy)    ──────────────────────┬──── 7 (run_eval) ──┬── 9 (stoch metrics)
3 (eval cfg)  ──────────────────────┤                    │
4 (imag_return) — regression guard  ├──── 8 (eval loop)  ├── 10 (e2e Habitat)
5 (ckpt roundtrip) ── 6 (resume)    │                    │
                                    └────────────────────┘
```

Cycles 1-5: independent, any order. Cycle 6 after 5. Cycle 7 after 2. Cycle 8 after 3+7. Cycle 9 after 7. Cycle 10 after 2+7+9.
