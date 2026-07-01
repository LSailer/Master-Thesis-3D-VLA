# AGENTS.md — `tests/r2dreamer/`

R2Dreamer test-suite rules. Inherits repo-root `AGENTS.md`; system under test is
`src/r2dreamer/`.

## Purpose

Tests cover agent init/act/train_step, RSSM and encoder shapes, optimiser,
checkpoint round-trips, launcher registries/presets/shims, video utilities, hybrid
encoder behavior, and JAX↔PyTorch numerical equivalence. Most tests are CPU-safe;
only real VGGT/Habitat-style integration paths need GPU.

## Running

```bash
uv run pytest tests/r2dreamer/ -m "not gpu" -q
uv run pytest tests/r2dreamer/test_agent.py -v
uv run pytest tests/r2dreamer/launch/test_presets.py

srun --partition=dev_gpu_h100 --gres=gpu:1 --time=00:30:00 \
  uv run pytest tests/r2dreamer/ -m gpu -v

./scripts/r2dreamer/run_decoder_probe_overfit_gpu.sh -v
```

## Contracts and gotchas

- `conftest.py` handles markers only. Shared fakes/factories belong in
  `conftest.py` or `tests/r2dreamer/_helpers.py`, not copy-pasted per file.
- `@pytest.mark.gpu` covers the real VGGT encoder and
  `world_model/test_decoder_probe_overfit_gpu.py`; run these through `srun`.
- CPU-safe VGGT tests use monkeypatched fake extractors with `.extract()`,
  `.reset()`, and `aggregator_feature_shape`.
- `test_cross_framework.py` imports `torch` and `external/r2dreamer/`; skip with
  `-k "not cross_framework"` where torch is unavailable.
- Real Habitat-Sim is not unit-tested here. Registry/curriculum/preset resolution
  is covered; full environment smoke belongs in sbatch/srun runs.
- Keep `launch/test_presets.py` in sync when adding encoders or Habitat curricula.
  `test_shim_invocation.py` derives run ids from `RUN_CONFIGS`; update its
  standalone shim list only for non-dispatcher `eval_*` entrypoints.
- JAX RNG keys are fixed integers for determinism.
