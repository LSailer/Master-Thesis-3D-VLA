# SOLID / decoupling refactor — 2026-07-06

Multi-agent structural refactor of the training codebase applying Single
Responsibility, Dependency Inversion (via `typing.Protocol`, no ABCs), and
pydantic v2 config validation. Behavior-preserving: full test suite matches
the pre-refactor baseline exactly (400 passed, 53 skipped, 11 environmental
failures from missing `data/curriculum/*.json` in the worktree — identical
list before and after).

Branch: `claude/vigorous-cannon-66c8ac` (uncommitted working tree at time of
writing). Scope: `src/r2dreamer/`, `src/configs/`, `src/buffer/`.
Excluded deliberately: `src/vggt/` (parity-critical JAX port of an external
architecture; bit-comparability with PyTorch takes precedence over SRP),
`src/vggt/reference/` (wraps external `StreamVGGT`), `archiv/`, `src/prototyp/`
(gitignored scratch).

## What changed

### 1. Encoder dispatch: three copies → one registry
Previously `agent.py` (`_resolve_encoder_cls`/`_make_encoder` + 8 helpers),
`observation_preparation/module_factory.py`, and
`observation_preparation/contracts.py` each hand-maintained the
`encoder_type` string → class/kwargs mapping.

New single source of truth: **`src/r2dreamer/encoders/registry.py`** —
`EncoderRegistryEntry(module_cls, kwargs_from_config, validate_config,
dummy_obs, diagnostics, direct_kwargs_from_config)`, resolved with an
MRO walk so encoder subclasses (e.g. GNN variants of
`HousePointsCameraEncoder`) inherit entries. Also defines the
`EncoderModule` Protocol. `module_factory.py` and `contracts.py` now
delegate to it. Placed under `encoders/` (not `launch/registries.py`) to
keep the dependency direction model-layer ← launcher-layer.

Two pre-existing kwargs divergences between the direct-construction path
and the contract-snapshot path (`TokenTransformerEncoder` `compute_dtype`;
`HouseGlobalEmbeddingEncoder` `token_dim`/`num_patch_tokens`) were
**preserved and documented inline**, not fixed — behavior first.

Adding a new encoder now means: one registry entry (including its optional
loss-diagnostics hook) instead of three synchronized edits plus an
`if encoder_type in (...)` branch in the loss.

### 2. `R2DreamerAgent` god class split (agent.py 1171 → 586 lines)
- `agent_modules.py` — `AgentModules` NamedTuple + `build_agent_modules()`:
  Flax module construction/init/param bundling.
- `agent_optim.py` — `make_optimizer()` (LaProp + AGC).
- `agent_loss.py` — `compose_agent_loss()` (former `_loss_fn`); the
  encoder-identity branch for hybrid-contribution metrics became a
  registry-resolved `encoder_loss_diagnostics()` hook (no-op default).
- `agent.py` remains the facade; public API and attribute names unchanged.

### 3. `Trainer` god class split (trainer.py 1053 → 797 lines)
- `replay_packing.py` — `replay_batch_to_arrays` + helpers (moved verbatim,
  re-exported from trainer for compat).
- `reporting.py` — `MetricsLogger` (CSV/W&B/timing/recon-image logging) and
  `EpisodeRecorder` (video/topdown capture), injected as optional
  constructor kwargs; defaults reproduce prior behavior including the
  wandb-disabled path. Loop logic (prefill/train/validate) stays in Trainer.

### 4. Adapters decoupled (`adapters/hybrid_adapter.py`, 781 → 1087 lines)
- `VGGTHousePointsPoseObsAdapter` is a thin coordinator over:
  `SceneBufferManager` (per-scene buffer lifecycle), `InputSubsamplingPolicy`,
  `HouseBufferDiagnostics`. The concrete `HouseContextPoseBuffer` is no
  longer instantiated inline — injected via `BufferFactory` callable typed
  against the `HouseContextPoseBufferLike` Protocol (default factory
  preserves behavior).
- `PointCloudDumper` extracted from `VGGTHouseGlobalEmbeddingObsAdapter`
  (PLY debug dumps out of the data path).
- `VGGTHouseContextObsAdapter` types its transformer param against
  `TokenContextEncoderLike` Protocol with a `default_context_transformer()`
  factory.
- Back-compat shims kept for private attrs used by tests/profiling scripts
  (e.g. `_house_context_snapshot` stays a bound method — monkeypatched by
  `scripts/profiling/profile_house_points_pose.py`).

### 5. Buffer kernel isolated (`src/buffer/`)
- `voxel_hash.py` (new) — pure stateless JAX voxel-hash kernel
  (`VoxelContextConfig`, `VoxelContextState`, `empty_state`,
  `add_frame_to_state` (jitted, donated), `house_context_snapshot`), moved
  byte-for-byte. Independently testable/reusable.
- `house_context_pose_buffer.py` keeps scene/dtype/PLY bookkeeping and
  re-exports the old private names. Constructor kwargs validated once at
  init via a frozen pydantic model (re-raised as `ValueError` to preserve
  the public contract); nothing added to jitted paths.

### 6. Configs → pydantic v2 (`src/configs/`)
`R2DreamerConfig`, `TrainerConfig`, `R2DreamerInterfaceConfig`,
`ObservationDims`, `ReplayObservationConfig`, `ObservationRunConfig` are now
`BaseModel`s (frozen where the dataclass was frozen) with
`extra="forbid"` — unknown/typo'd kwargs now fail at construction instead
of vanishing silently. The `ReplayObservationConfig.__post_init__`
validator became `@model_validator(mode="after")` (pydantic
`ValidationError` is a `ValueError` subclass — exception contract intact).
`checkpointing.config_snapshot` gained a `BaseModel` branch
(`model_dump()`); snapshot JSON verified byte-identical (tuples preserved,
`encoder_module_cls` popped) and the save/load resume round-trip passes.

Deliberately **no** positivity constraints: production SLURM configs use
legitimate zeros (`mlp_layers: 0` bare-linear readout, `val_every: 0`).
`observation_preparation/contracts.py` deliberately stays dataclass-based:
its hand-tuned `to_snapshot`/`from_snapshot` controls checkpoint JSON and
carries live `type[nn.Module]` objects — pydantic would add serializer risk
for no benefit.

### 7. VGGT extractor seam (`encoders/base.py`)
`VGGTExtractorFactory` Protocol (typed on the *factory* — the encoder only
constructs the extractor, never calls it) + `extractor_factory` constructor
kwarg / overridable class attribute. Default reproduces the lazy
`import_module("src.vggt.jax.feature_extractor")` exactly, so CPU-only
environments still never import GPU deps. This was the one DIP violation
crossing the r2dreamer→vggt boundary.

### 8. Launch god-modules split (`src/r2dreamer/launch/`)
- `eval_cli.py` — composition root: checkpoint/manifest resolution,
  concrete env/encoder/agent construction (concrete classes belong here).
- `eval_loop.py` — `EvalEnv`/`EvalAdapter`/`EvalAgent` Protocols + pure
  rollout helpers.
- `eval_artifacts.py` — video/topdown/JSON writing, W&B lifecycle.
- `settings.py` — the CLI-vs-shim precedence helpers previously duplicated
  between train and evaluate.
- `agent_overrides.py` — typed CLI→config bridge: mapping tables
  (`_SCALAR_OVERRIDES` split pre/post latent-preset to preserve override
  ordering, `_FLAG_OVERRIDES` for inverted bools, dtype aliases) validated
  against `R2DreamerConfig.model_fields` at construction. A renamed/removed
  config field now raises naming the offending `--flag → 'field'` pair
  instead of silently dropping the override.
- `evaluate.py` (548 → 351) and `train.py` (375 → 341) stay as thin entry
  points; all names tests monkeypatch remain importable from their old
  locations. Entry points (`src.main`, SLURM shims) verified via `--help`
  smoke runs.

## Net stats
16 files modified (−1964 / +948 lines), 12 new single-purpose modules.
All public import paths preserved via re-exports.

## Verification protocol used
1. Baseline: full CPU suite before any edit → 400 passed / 53 skipped /
   11 failed (all missing-curriculum-data, environmental).
2. Each refactor agent ran its module's targeted tests + a broad sweep;
   pre-existing failures were cross-checked against the baseline list
   (one agent proved non-causation by `git stash` + re-run).
3. Final: identical full-suite command, diffed line-by-line against the
   baseline output — only the timing line differs.

Pre-existing issues NOT addressed (out of scope, pre-dating refactor):
`tests/r2dreamer/test_cross_framework.py` collection error (missing `rssm`
module), `tests/environments/test_spl.py` collection error, and the 11
curriculum-data failures (fixable by syncing `data/curriculum/` into the
worktree).

## Adversarial review→fix loop (post-refactor)

An adversarial reviewer (briefed to argue the refactor was NOT finished,
with file:line evidence required) ran against the result, alternating with
a fixing agent. Capped at 5 fix iterations; terminated after 2 when the
reviewer explicitly conceded no CRITICAL/MAJOR finding remained.

**Round 1 findings (all fixed):**
- F1 MAJOR — the "RGB-bearing encoder_type" concept was still triplicated
  as hand-synced frozensets in `module_factory.py`, `agent_modules.py`, and
  `decoder_targets.py`, contradicting the "one edit per new encoder" claim.
  → consolidated into `encoders/registry.py:RGB_BEARING_ENCODER_TYPES`
  (+ `encoder_type_has_rgb_target()`); lazy import in `decoder_targets.py`
  breaks a registry→…→decoder_targets cycle (trace-time only, not hot-path).
- F2 MAJOR — new modules had zero direct unit tests. → 23 new tests:
  `tests/r2dreamer/encoders/test_registry.py` (MRO walk, direct-vs-snapshot
  kwargs divergence, F1 consistency lock), `test_reporting.py`,
  `tests/buffer/test_voxel_hash.py`, `tests/r2dreamer/launch/test_settings.py`,
  `test_agent_overrides.py`.
- F3 MAJOR — hybrid_adapter collaborators all lived in one 1087-line file.
  → moved to `adapters/scene_buffer.py`, `subsampling.py`,
  `house_diagnostics.py`, `point_cloud_dumper.py`; re-exports keep import
  paths; module docstring rewritten.
- F4 MINOR — Trainer still owned the CSV file/writer lifecycle. →
  `MetricsLogger.open_csv(path, resume)` context manager; trainer threads
  the logger, not raw `(writer, f)` handles.
- F5 MINOR — stale non-frozen justification comment in agent_config.py
  corrected (real reason: `encoder_input_contract` reassignment in train.py).
- F6 MINOR — docstring gaps (`VoxelContextConfig` attributes, MetricsLogger
  API) filled.

**Round 2:** the reviewer verified every round-1 fix against git HEAD
(per-row CSV `flush()` crash-recovery preserved; file closed on exception;
resume/header semantics line-identical; re-exports complete; snapshot
JSON still tuple-faithful) and conceded — no CRITICAL/MAJOR. Two MINORs,
both fixed:
- F7 — overfit-mode first-row `perf/fps_interval` was 0 at HEAD only via an
  unset-attribute accident; the refactor now reports a real measured
  interval. Accepted as intended (diagnostic-only) and pinned with a
  regression test.
- F8 — `MetricsLogger` row-writing methods now raise a clear
  `RuntimeError` outside an `open_csv` context; added tests for per-row
  flush persistence, outside-context raise, and handle release on
  exception.

Final state: 27 new test functions (35 cases with parametrization) from
the loop; full suite 435 passed / 53 skipped / 11 failed — failure list
character-identical to the pre-refactor baseline (400 passed then; the
+35 are the loop's new tests).

## Pattern references
- Real Python — SOLID in Python: https://realpython.com/solid-principles-python/
- ArjanCodes — Dependency Inversion in Python: https://arjancodes.com/blog/dependency-inversion-principle-in-python-programming/
- Duck typing & DIP with `typing.Protocol`: https://levelup.gitconnected.com/duck-typing-and-dependency-inversion-in-python-f19ffac48099
- Pydantic v2 models / settings: https://docs.pydantic.dev/latest/concepts/models/ , https://docs.pydantic.dev/latest/concepts/pydantic_settings/
