# AGENTS.md — `src/r2dreamer/`

R2Dreamer package rules. Inherits repo-root `AGENTS.md`.

## Purpose

R2Dreamer is a JAX/Flax DreamerV3-style world-model agent for Habitat ObjectNav
and Crafter. This package contains the library code; runnable drivers live in
`scripts/r2dreamer/`, tests in `tests/r2dreamer/`.

## Where to look

- `agent.py`: module bundle, optimiser, slow critic, and the two directly
  jitted entry points `act()`/`train_step()`.
  The composition root itself is `src/main.py`.
- `src/configs/agent_config.py`: `R2DreamerConfig`, size presets, hyperparameters.
- `world_model/`: RSSM, prediction heads, world-model loss, `rssm_factory.py`.
- `behavior/`: imagination, lambda returns, actor/critic losses.
- `representation/`: Barlow Twins + replay-value losses.
- `encoders/`: the Flax branch modules; `routed_composite.py` composes one
  branch per routed adapter field.
- `experience.py`: env loop -> adapter -> replay collector.
- `src/adapters/`: per-variant observation adapters; each field declares its
  encoder branch, whether it is replayed, and whether it is the decoder target.

## Contracts

- **Routing, not strings:** the architecture comes from one live adapter call
  (`AdapterField.encoder`), never from a config string. `R2DreamerConfig.adapter`
  is provenance only.
- **Config-first:** `R2DreamerConfig` owns everything else; CLI flags override it.
- **Observation layout:** images are HWC and normalized inside `ConvEncoder`;
  metric/feature fields must not be `/255` normalized.
- **Params:** plain dict pytree keyed by module groups (`encoder`, `rssm`, `actor`, etc.).
- **JIT/PRNG:** `act` and `train_step` are directly jitted with static `self`
  (hashed by identity, traced once per instance), so their bodies read
  architecture off `self` but never mutable state: params, the acting carry and
  the train state are arguments and return values. Callers thread the carry and
  reassign `agent.train_state`; `train_step` metrics are device arrays,
  converted at the log sites via `materialize_metrics` (`materialize=False`
  returns `{}`). The `agent.py` docstrings own the full hazard. Always split
  JAX keys explicitly.
- **Checkpoints:** pickle params/opt/slow critic/EMA/step only. The encoder is
  rebuilt from the adapter routing at load time and `_assert_params_match`
  catches drift.
- **Episode boundaries:** `is_first` must be truthful; RSSM and `act()` reset on it.
- **KL:** DreamerV3 asymmetric KL; dynamics detaches posterior, representation detaches prior.
- **Imagination:** RSSM/reward/continue are used under `stop_gradient`; bootstrap from slow critic.
- **Actions:** replay stores int32; `ReplayBuffer.sample()` one-hots to `(B,T,A)` float.

## Adding an encoder/input mode

Add the adapter in `src/adapters/`, register it in `src/adapters/__init__.py`
(`ADAPTERS`), and route each field to an `Encoder` member. A branch that does
not exist yet goes in `encoders/` and is wired into
`encoders/routed_composite.py`. `tests/adapters/` covers every registered
variant automatically. To make it launchable, add a
`scripts/slurm/configs/*.yaml` whose `args:` name the adapter (and curriculum);
the variant reaches `python -m src.main` as `--adapter <name>`.

A variant that differs only in a constant (a coarser reduction, another cloud
branch, no appearance channel) is a subclass overriding that constant, not a
copy of the pipeline.

## Running/testing

GPU execution (`jax`, `habitat_sim`, VGGT) must run under `srun`. CPU-only tests:

```bash
uv run pytest tests/r2dreamer/ -m "not gpu" -q
```

GPU-marked tests:

```bash
srun --partition=dev_gpu_h100 --gres=gpu:1 --time=00:30:00 \
  uv run pytest tests/r2dreamer/ -m gpu
```
