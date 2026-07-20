## Why

Config→Flax-module-constructor kwargs are resolved in **two duplicate places** that
must be hand-kept in sync: the `encoder_module_kwargs_from_config` per-class table in
`src/r2dreamer/observation_preparation/contracts.py` (dispatching by module-class name
via `_kwargs_dispatch_name`'s "nearest handled ancestor" heuristic) and the per-class
`_make_*` kwargs builders in `src/r2dreamer/encoders/factory.py`. This desync just
caused 12 test failures (Cause A): the `contracts.py` table still emits
`embed_dim`/`token_dim`/`num_patch_tokens`/`reducer_hidden`/`reducer_layers`/
`camera_hidden`/`camera_layers`/`rgb_branch` for `vggt_house_global_embedding` after the
`HouseGlobalEmbeddingEncoder.__init__` signature drifted, so construction fails with
`unexpected keyword argument 'embed_dim'`. The two tables are a structural footgun: a
module signature change can silently outrun the faraway kwargs table.

## What Changes

- Add a `module_kwargs_from_config(cls, config) -> dict` **classmethod** on the launcher
  `Encoder` class — the single source of truth, defined next to its `module_cls`, for how
  a config maps to that encoder's Flax-module constructor kwargs.
- `Encoder` base provides a default (empty / `NotImplementedError`-on-missing-`module_cls`);
  `VGGTEncoder` reads kwargs from the variant spec for variant-driven encoders; standalone
  encoders (`VGGTHouseGlobalEmbeddingEncoder`, `VGGTHousePointsPoseEncoder`,
  `VGGTHybridHousePointsPoseEncoder`, the GNN/PointNet variants, `HybridEncoder`,
  `CNNEncoder`) each define their own.
- The three call sites (`launch/train.py:202`, `checkpointing.py:84`, `factory.py`
  `_make_*`) delegate through `encoder_registry[config.encoder_type].module_kwargs_from_config(config)`
  instead of dispatching by module-class name.
- `encoder_module_kwargs_from_config` in `contracts.py` becomes a **one-line shim** that
  does the registry lookup — kept as the stable entry point so callers don't all import
  `encoder_registry` and so the `(config, module_cls)` signature stays backward-compatible.
- **BREAKING (internal):** the per-class kwargs table in `contracts.py`
  (`_kwargs_dispatch_name` + the `if class_name == ...` ladder) is removed; the `_make_*`
  kwargs-building bodies in `factory.py` are removed (the functions either delegate or are
  deleted). No public API changes — `encoder_module_kwargs_from_config` keeps its signature.
- The `test_module_constructs_from_contract_kwargs` regression test becomes a check that
  each registry encoder's `module_kwargs_from_config` constructs its own `module_cls`.

## Capabilities

### New Capabilities
- `encoder-module-kwargs`: A launcher `Encoder` owns the single source of truth for
  deriving its Flax module's constructor kwargs from an effective config; the contract
  snapshot, train/eval, and the agent module factory all delegate to it instead of
  consulting a separate module-class-keyed kwargs table.

### Modified Capabilities
<!-- None: openspec/specs/ is empty; there are no existing requirements to amend. -->

## Impact

- **Code**: `src/r2dreamer/encoders/base.py` (new classmethod on `Encoder` + `VGGTEncoder`),
  each standalone encoder module (`house_global_embedding.py`, `house_points_pose.py`,
  `gnn_house.py`, `pointnet.py`, `__init__.py` hybrid/house-context, `cnn.py`),
  `src/r2dreamer/encoders/factory.py` (delete `_make_*` kwargs bodies / `_contract_encoder_kwargs`,
  delegate), `src/r2dreamer/observation_preparation/contracts.py` (delete the per-class table,
  shrink `encoder_module_kwargs_from_config` to a shim), `src/r2dreamer/checkpointing.py` and
  `src/r2dreamer/launch/{train,evaluate}.py` (unchanged call sites — they already call the
  shim, which now delegates).
- **Tests**: `tests/r2dreamer/launch/test_registries.py::test_module_constructs_from_contract_kwargs`
  stays valid (it constructs via the same shim) and goes green once each encoder's kwargs match
  its module signature. Resolves the `vggt_house_global_embedding` Cause-A failure **only after**
  the separate rich-API restoration makes `HouseGlobalEmbeddingEncoder` accept those kwargs again
  (out of scope here; this change assumes the signature already accepts them).
- **Risk**: re-couples fresh-build to the `encoder_type` registry key (eval-from-checkpoint is
  unaffected — it uses the stored `encoder_module_kwargs` snapshot, not the resolver). Config
  field-name knowledge (`vggt_embed_dim`, `mlp_vggt_hidden`, …) moves from one central table
  into ~17 encoder files.
- **Out of scope**: the `HouseGlobalEmbeddingEncoder` rich-API restoration, the stale
  `encode_house_global_obs` import in `tests/r2dreamer/launch/test_encoders.py`, and all
  environment errors (habitat/omegaconf/ruamel/curriculum-data/pose-buffer).