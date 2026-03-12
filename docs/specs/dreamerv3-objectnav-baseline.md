# DreamerV3 ObjectNav Baseline Training Spec

## 1. Overview

Train DreamerV3 from pixels on HM3D ObjectNav. Measure success rate + SPL.
Baseline for thesis — later compare with UNITE 3D feature injection.

## 2. Environment

**Primary: HM3D ObjectNav (`src/dreamerv3/env_habitat.py`)**
- 4 discrete actions: STOP, FORWARD, LEFT, RIGHT
- RGB 256x256, 500-step episodes
- HM3D v2: 80 train scenes, 20 val scenes, 6 object categories

**Future work: AI2-THOR / ProcTHOR**
- ProcTHOR: procedural 10k+ houses, richer object diversity
- Would need `env_ai2thor.py` mirroring `env_habitat.py` interface
- Benefit: unlimited scene diversity for WM generalization
- *Not in scope — separate issue*

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

**Ablation alternatives:**

| Reward | Formula | Pros | Cons |
|--------|---------|------|------|
| Sparse | `10.0 * success` | Clean signal | Very sparse, slow |
| Geodesic delta | `Δd_geo + 10*success` | Dense, path-aware | Needs Habitat geodesic API |
| Euclidean delta | `Δd_eucl + 10*success` | Simple | Ignores walls |
| Slack penalty | `Δd_geo - 0.01 + 10*success` | Anti-dawdling | Tuning needed |

**Plan:** geodesic_delta default, sparse as ablation.

## 5. Training Pipeline

**Existing flow (`src/dreamerv3/train.py`, no major changes):**
1. Prefill 5k random steps → replay buffer
2. act → env.step → buffer.add → sample batch → train WM + actor-critic
3. Batch: 16 seq x 64 timesteps
4. Imagination: 15-step rollouts
5. Checkpoint every 50k steps, log to W&B

**Compute plan (48h single H100):**
- 5M env steps target (~10k episodes)
- **Step 0: profiling run** — 10k steps to measure actual steps/min, then confirm 5M fits in 48h
- 1 seed first to validate, scale to 3 seeds if promising

**Metrics:**
- Primary: `success_rate`, `spl`
- Episode: `episode_reward`, `episode_length`
- WM: `recon_loss`, `kl_loss`, `reward_loss`, `cont_loss`
- Behavior: `actor_loss`, `critic_loss`, `entropy`, `imag_return`

## 6. Evaluation Protocol

- Eval on `val` split every 100k steps (no training)
- **Both** greedy (argmax) and stochastic policy logged; greedy = primary metric
- Report: success rate, SPL, distance-to-goal at termination
- Target: ~10-20% success (DreamerV3 from pixels is hard; DD-PPO w/ depth+GPS ≈ 30-40%)

## 7. Coding Agent Done-Criteria

Concrete, machine-verifiable checks a coding agent on BWUniCluster can run.

### Gate 1: Static checks (no GPU, no Habitat needed)

```bash
# all must exit 0
uv run pytest tests/test_dreamerv3_shapes.py -v          # 16 shape tests
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
