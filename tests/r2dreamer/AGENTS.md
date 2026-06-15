# AGENTS.md — `tests/r2dreamer/`

Module contract for the R2Dreamer test suite. Scopes the repo-root
[`AGENTS.md`](../../AGENTS.md) to this folder. System under test:
[`src/r2dreamer/`](../../src/r2dreamer/AGENTS.md).

## Purpose

Validates the R2Dreamer JAX/Flax agent end to end: agent init / `act` / `train_step`,
RSSM and encoder shapes, the LaProp+AGC optimiser, checkpoint round-trips, launcher
registries/presets/shims, video utilities, the hybrid encoder, and **JAX↔PyTorch
numerical equivalence**. Most tests are CPU-only by design; GPU is needed only for the
real VGGT extractor.

## File map

### Root (`tests/r2dreamer/`)
| File | Covers | GPU |
|------|--------|-----|
| `test_agent.py` | init, `act`, `train_step` metrics, deterministic update, loss composition | CPU |
| `test_optim.py` | `laprop`, `agc` (`src.shared.optim`) | CPU |
| `test_shapes.py` | `R2DreamerConfig` shape derivation; `RMSNorm`, `BlockLinear`, `Deter`, `R2RSSM` | CPU |
| `test_trainer.py` | `convert_batch`, `save/load_checkpoint`, resume | CPU |
| `test_vggt_encoder.py` | VGGT encoders + `VGGTReplayBuffer` sampling / `is_first` (monkeypatched extractor) | CPU |
| `test_cross_framework.py` | torch↔JAX weight transfer + equivalence (RMSNorm, BlockLinear, encoder, RSSM observe/prior, KL, Barlow, reward head, λ-return) | CPU, needs `torch` |

> The offline-pipeline tests (`test_collect_offline_buffer.py`,
> `test_external_offline_buffer.py`, `test_offline_comparison.py`, and
> `tests/buffer/test_offline_buffer_dataset.py`) were archived with the offline
> pipeline to `archiv/offline-r2dreamer-20260602/` (3D-25/26/45/46) and are no
> longer collected.

### `launch/`
| File | Covers | GPU |
|------|--------|-----|
| `test_encoders.py` | encoder construction/specs/adapters; `TestVGGTEncoder` loads the real model | **GPU (that class)** |
| `test_registries.py` | `observation_preparation_registry`, `env_registry`, `CURRICULA` entries | CPU |
| `test_presets.py` | `(env, encoder, curriculum)` preset matrix resolves (parametrized) | CPU |
| `test_parser.py` | train parser does not expose `wandb_notes_file` | CPU |
| `test_shim_invocation.py` | every `run.py <run-id>` (parametrized over `RUN_CONFIGS`) + each standalone `eval_*`/validation shim `--help` exits 0 (subprocess) | CPU |
| `test_video_utils.py` | `compose_frame`, `render_topdown_frame`, `log_episode_video` | CPU |
| `fixtures/` | `sample_habitat_obs.npz` (~3.3 MB), `expected_vggt_outputs.npz` (~150 KB) | — |

### `world_model/`
| File | Covers | GPU |
|------|--------|-----|
| `test_hybrid_encoder.py` | hybrid encoder+decoder (3D-50/51/52): gate / vggt / cnn branches, decoder round-trip | CPU |

## How to run

```bash
uv run pytest tests/r2dreamer/ -m "not gpu" -q          # full CPU suite
uv run pytest tests/r2dreamer/test_agent.py -v          # one file
uv run pytest tests/r2dreamer/launch/test_presets.py    # parametrized matrix

# GPU tests must be wrapped in srun (see root AGENTS.md):
srun --partition=dev_gpu_h100 --gres=gpu:1 --time=00:30:00 \
  uv run pytest tests/r2dreamer/ -m gpu -v
```

CI (`.github/workflows/ralph.yml`) runs `uv run pytest tests/<file> -x -q` and wraps GPU
selections with the same `srun` invocation.

## Conventions

- **`conftest.py` (repo root) handles markers only** — it skips `@pytest.mark.gpu`
  tests without a JAX GPU backend and gates the VGGT parity test behind
  `RUN_VGGT_PARITY=1`. Per-test fixtures (`cfg`, `agent`, `rng`, `replay_batch`) are
  still defined per-file today; cross-file fakes/factories (e.g. a shared
  `FakeExtractor`, a minimal-config builder) belong in `conftest.py` or a
  `tests/r2dreamer/_helpers.py`, not copy-pasted per file. JAX RNG keys are fixed
  integers for determinism (e.g. init=7, train=11).
- **Markers:** `@pytest.mark.gpu` (only `test_encoders.py::TestVGGTEncoder`),
  plus `habitat_sim` / `integration` registered but currently unused.
- **Mocking:** `monkeypatch` + hand-rolled fake extractors (`.extract()/.reset()/
  aggregator_feature_shape`) keep VGGT tests CPU-safe. Pytree comparison via
  `tree_allclose` / `tree_any_changed`; cross-framework tolerances `ATOL_COMPONENT=1e-4`,
  `ATOL_COMPOSED=2e-3`.
- **Parametrization:** preset matrix (`test_presets.py`), shim list (`test_shim_invocation.py`),
  `unimix` (`test_cross_framework.py`).

## Gotchas / read-this-first

- **`test_cross_framework.py` imports `torch`** and the `external/r2dreamer/` reference. It
  lives in the default suite; skip with `-k "not cross_framework"` where torch is absent.
- **Real Habitat-Sim is not unit-tested** — only registry/curriculum/preset *resolution* is.
  Full env instantiation is deferred to sbatch smoke runs.
- **Keep these lists in sync with the code:** the `PRESETS` matrix in `test_presets.py` must be
  updated when you add an encoder or curriculum. `test_shim_invocation.py` derives its run ids
  from `RUN_CONFIGS`, so new runs are covered automatically; update its `STANDALONE_SHIMS` list
  only when adding a non-dispatcher (`eval_*` / validation) entrypoint.
- **`TestVGGTEncoder` downloads + loads the real VGGT model** and needs GPU memory; run it via
  `srun`, not bare `pytest`.
