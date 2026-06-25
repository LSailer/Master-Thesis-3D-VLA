# AGENTS.md — `src/r2dreamer/`

R2Dreamer package rules. Inherits repo-root `AGENTS.md`.

## Purpose

R2Dreamer is a JAX/Flax DreamerV3-style world-model agent for Habitat ObjectNav
and Crafter. This package contains the library code; runnable drivers live in
`scripts/r2dreamer/`, tests in `tests/r2dreamer/`.

## Where to look

- `agent.py`: composition root; params, optimiser, slow critic, JIT `train_step()`/`act()`.
- `config.py`: `R2DreamerConfig`, size presets, hyperparameters.
- `trainer.py`: env loop, prefill, replay conversion, logging, checkpoints.
- `world_model/`: RSSM, encoders, heads, world-model loss.
- `behavior/`: imagination, lambda returns, actor/critic losses.
- `representation/`: Barlow Twins + replay-value losses.
- `adapters/`: env observation to replay/agent bridge.
- `observation_preparation/`: encoder-input contracts and VGGT readouts.
- `launch/`: train/evaluate entrypoints, parser, registries, Habitat factory.

## Contracts

- **Config-first:** `R2DreamerConfig` is the source of truth; CLI flags override it.
- **Observation layout:** RGB is CHW and normalized in `ConvEncoder`; VGGT/hybrid values
  are metric/features and must not be `/255` normalized.
- **Params:** plain dict pytree keyed by module groups (`encoder`, `rssm`, `actor`, etc.).
- **JIT/PRNG:** `train_step` and `act` are JIT-compiled; always split JAX keys explicitly.
- **Checkpoints:** pickle params/opt/slow critic/EMA/step plus optional
  `encoder_input_contract` JSON snapshot.
- **Episode boundaries:** `is_first` must be truthful; RSSM and `act()` reset on it.
- **Encoder wiring:** encoder, adapter, replay fields, and `obs_shape` must agree.
- **KL:** DreamerV3 asymmetric KL; dynamics detaches posterior, representation detaches prior.
- **Imagination:** RSSM/reward/continue are used under `stop_gradient`; bootstrap from slow critic.
- **Actions:** replay stores int32; `convert_batch()` one-hots to `(B,T,A)` float.

## Adding encoder/input modes

Implement the encoder/spec in `encoders/__init__.py`, register it in
`launch/registries.py`, add the run preset in `scripts/r2dreamer/_run_configs.py`,
and cover it in `tests/r2dreamer/launch/test_presets.py`.

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
