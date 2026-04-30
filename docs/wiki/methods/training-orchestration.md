# Training Orchestration Module

**Status**: implemented
**Date**: 2026-04-15
**Tags**: #refactor #trainer #deep-module
**GitHub Issue**: #68

---

## Motivation

Three training scripts (`run_jax_crafter.py`, `run_jax_habitat.py`, `run_jax_habitat_vggt.py`) were ~70% copy-pasted. Duplicated concerns: `_convert_batch()` (3 copies), `_save_checkpoint()` (2 copies, both missing `ema_state`), two replay buffer classes, training loop logic. Total: 758 lines of scripts with high coupling.

## What Changed

### Unified ReplayBuffer (`modules/dreamerv3/replay_buffer.py`)

`ReplayBuffer` + `VGGTReplayBuffer` merged into one class parameterized by `BufferConfig`:
- `obs_dtype="uint8"` + `normalize_obs=True` = old image buffer behavior
- `obs_dtype="float32"` + `normalize_obs=False` = old VGGT buffer behavior

Backward-compat shim accepts `DreamerConfig`/`R2DreamerConfig` directly.

### Trainer Module (`modules/r2dreamer/trainer.py`)

| Component | Purpose |
|---|---|
| `convert_batch()` | Replay buffer output -> agent input (one-hot actions, field renaming) |
| `save_checkpoint()` / `load_checkpoint()` | Full agent state serialization including `ema_state` |
| `ObsAdapter` | Bridges env obs to buffer/agent; subclass `VGGTObsAdapter` for VGGT |
| `TrainerConfig` | Loop control (steps, logging, checkpointing, WandB, val loss) |
| `habitat_defaults()` | Pre-configured adapter + episode metrics for Habitat+CNN |
| `Trainer` | Prefill -> train (train-ratio) -> log -> checkpoint loop |

### Script Reduction

| Script | Before | After |
|---|---|---|
| `run_jax_crafter.py` | 139 lines | 56 lines |
| `run_jax_habitat.py` | 302 lines | 101 lines |
| `run_jax_habitat_vggt.py` | 317 lines | 134 lines |
| **Total** | **758** | **291 (-62%)** |

## Bug Fixes

- `save_checkpoint()` now includes `agent.ema_state` (return normalization). Old code silently dropped it, causing return normalization to reset on resume.

## Tests

- `test_replay_buffer.py`: 12 tests covering uint8, float32, normalization, wrapping, is_first
- `test_trainer.py`: 8 tests for convert_batch (5) and checkpoint round-trip (3, including ema_state)

## Related

- [[world-model-training-loop]] — explains the training loop structure this module encapsulates
