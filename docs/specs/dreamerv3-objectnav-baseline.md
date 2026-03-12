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

## 7. Coding Agent Done-Criteria

Concrete, machine-verifiable checks a coding agent on BWUniCluster can run.

### Gate 1: Static checks (no GPU, no Habitat needed)

```bash
# all must exit 0
uv run pytest tests/test_dreamerv3_shapes.py -v          # 9 shape tests
uv run mypy src/dreamerv3/ --ignore-missing-imports       # type check
uv run ruff check src/dreamerv3/                          # lint
uv run ruff format --check src/dreamerv3/                 # format
```

### Gate 2: Integration smoke test (needs Habitat + GPU)

```bash
# runs 100 env steps, trains 1 batch, exits cleanly
uv run python -m src.dreamerv3.train \
  --total_steps 100 --prefill_steps 50 --split val_mini \
  --logdir /tmp/dreamer_smoke_$$
# verify checkpoint exists
test -f /tmp/dreamer_smoke_$$/checkpoint.pkl
```

**Pass condition:** exit code 0, no OOM, no NaN in printed losses.

### Gate 3: Profiling run (10k steps, ~30 min)

```bash
uv run python -m src.dreamerv3.train \
  --total_steps 10000 --prefill_steps 1000 \
  --logdir output/dreamerv3/profile-run
```

**Pass conditions (agent must verify from stdout/W&B):**
- Throughput printed or logged (steps/sec)
- `recon_loss` at step 10k < `recon_loss` at step 1k (WM learning)
- No NaN in any logged metric
- Extrapolated time for 5M steps fits in 48h (steps/sec x 5M < 172800 sec)

### Gate 4: Eval loop works

```bash
# eval mode: agent acts greedy on val split, logs success/SPL
uv run python -m src.dreamerv3.train \
  --total_steps 200 --prefill_steps 50 --split val_mini \
  --eval_every 100 --eval_episodes 5 \
  --logdir /tmp/dreamer_eval_$$
```

**Pass condition:** W&B (or stdout) shows `eval/success_rate_greedy` and `eval/success_rate_stochastic` keys.

**Implementation notes (eval loop):**
- Add `eval_every: int = 100_000` and `eval_episodes: int = 10` to `DreamerConfig` (`configs.py`)
- Create separate `HabitatObjectNavEnv(split="val_mini")` at startup for eval
- Every `eval_every` steps, run `eval_episodes` on val env
- RSSM state resets naturally per episode (`is_first=True` zeros state — already implemented)
- Greedy mode: wire up existing `training` param in `agent.act()` (currently unused) — `training=False` → `jnp.argmax(logits)` instead of `jax.random.categorical`
- Run both greedy + stochastic eval, log `eval/success_rate_greedy`, `eval/spl_greedy`, `eval/success_rate_stochastic`

### Gate 5: Full training launch (human triggers, agent verifies)

```bash
sbatch scripts/slurm/train_dreamer.sbatch
# agent checks after job starts:
squeue -u $USER -n dreamerv3  # job is RUNNING
```

**Post-training checks (agent runs after job completes):**
```bash
# checkpoint exists
test -f output/dreamerv3/run-*/checkpoint.pkl
# W&B run has >4M steps logged
# WM loss at final step < WM loss at step 50k (still learning or converged)
# success_rate > 0 at some point during training
```

### Gate 6: Results documented

- W&B project `dreamerv3-objectnav` contains ≥1 completed run
- Run has all metrics from §5 logged
- `docs/specs/dreamerv3-objectnav-baseline.md` exists and matches this spec

### Summary: agent autonomy levels

| Gate | Autonomous? | Notes |
|------|-------------|-------|
| 1 Static | Yes | run anytime, no GPU |
| 2 Smoke | Yes | needs Habitat env + GPU |
| 3 Profile | Yes | ~30 min, verify throughput |
| 4 Eval | Yes | needs code changes first (add eval loop) |
| 5 Full train | Human triggers | agent verifies post-completion |
| 6 Docs | Yes | create spec file |
