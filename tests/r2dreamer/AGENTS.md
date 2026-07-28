# AGENTS.md — `tests/r2dreamer/`

R2Dreamer test-suite rules. Inherits repo-root `AGENTS.md`; system under test is
`src/r2dreamer/`.

## Purpose

Tests cover agent init/act/train_step, RSSM shapes, the routed encoder branches
in `encoders/test_routed_branches.py`, optimiser, checkpoint round-trips, the
argparse surface and eval manifest under `launch/`, video utilities, and
JAX↔PyTorch numerical equivalence. Per-variant adapter routing (registry
coverage, replay round-trip) lives one level up in `tests/adapters/`, not here.
Most tests are CPU-safe; only real VGGT/Habitat-style integration paths need GPU.

## Running

```bash
uv run pytest tests/r2dreamer/ -m "not gpu" -q
uv run pytest tests/r2dreamer/test_agent.py -v
uv run pytest tests/r2dreamer/launch/test_shim_invocation.py

srun --partition=dev_gpu_h100 --gres=gpu:1 --time=00:30:00 \
  uv run pytest tests/r2dreamer/ -m gpu -v

./scripts/r2dreamer/run_decoder_probe_overfit_gpu.sh -v
```

## Contracts and gotchas

- The repo-root `conftest.py` handles markers/skips plus the autouse fixture
  that drops JAX's jit caches after any test that built an agent (static `self`
  makes JAX pin every instance and its executable for the session). It is
  load-bearing for memory, not cosmetic - see its docstring before touching it.
  Shared fakes and factories belong in a `conftest.py`, not copy-pasted per file.
- `@pytest.mark.gpu` covers `world_model/test_decoder_probe_overfit_gpu.py`; run
  it through `srun`.
- CPU-safe VGGT tests use a fake extractor exposing only `.extract(frame)`, the
  one method `src.adapters.contract.FeatureExtractor` requires; the shared stub
  is `tests/adapters/conftest.py::FakeExtractor`.
- `test_cross_framework.py` imports `torch` and `external/r2dreamer/`; skip with
  `-k "not cross_framework"` where torch is unavailable.
- Real Habitat-Sim is not unit-tested here. Curriculum/run-id resolution is
  covered; full environment smoke belongs in sbatch/srun runs.
- Adding a variant needs no edit here: `tests/adapters/` parametrizes over
  `src.adapters.ADAPTERS`. `launch/test_shim_invocation.py` derives run ids from
  `RUN_CONFIGS`; update its standalone list only for non-dispatcher `eval_*`
  entrypoints.
- JAX RNG keys are fixed integers for determinism.
