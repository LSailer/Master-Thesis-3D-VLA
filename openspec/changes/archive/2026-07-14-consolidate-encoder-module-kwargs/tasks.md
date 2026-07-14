## 1. Add `module_kwargs_from_config` to the launcher `Encoder` hierarchy

- [x] 1.1 Add `module_kwargs_from_config(cls, config) -> dict` classmethod on `Encoder` in `src/r2dreamer/encoders/base.py`. Base raises `NotImplementedError` (it carries no module) rather than returning an empty dict, so every registered subclass must define the formula explicitly.
- [x] 1.2 Override it on `VGGTEncoder`. Deviation from D2 as originally worded: the variant spec carries `module_cls` but no kwargs *values*, so `VGGTEncoder.module_kwargs_from_config` resolves the module's *identity* from `VGGT_VARIANTS[cls.variant_key].module_cls` and reads the *values* from `config` via the module-name dispatch in `_vggt_module_kwargs` (co-located helper in `base.py`). Bare variant subclasses inherit this unchanged.
- [x] 1.3 Override it on `CNNEncoder` to emit `depth`/`kernel_size`/`mults` only. The `vggt_wp_dense_cnn` `input_kind`/`embed_dim` extras are NOT emitted here — `vggt_wp_dense_cnn` is a VGGT variant (`VGGTDenseWPEncoder`), so its ConvEncoder extras live in the VGGT dispatch (`_vggt_module_kwargs`'s `ConvEncoder` branch), not on `CNNEncoder`.
- [x] 1.4 `HybridEncoder` does NOT override the method — it inherits `VGGTEncoder`'s dispatch (the `_vggt_module_kwargs` `HybridEncoder` branch covers it). No standalone override needed.
- [x] 1.5 Override it on `VGGTHousePointsPoseEncoder` (house-points/pose kwargs incl. `house_point_norm`) and `VGGTHybridHousePointsPoseEncoder` (calls `super().module_kwargs_from_config(config)` then adds the `cnn_depth`/`cnn_kernel`/`cnn_mults` hybrid knobs).
- [x] 1.6 The GNN variants (`GnnHousePointsPoseEncoder`, `GnnEdgeHousePointsPoseEncoder`) and `PointNetHousePointsPoseEncoder` do NOT override the method — they inherit `VGGTHousePointsPoseEncoder`'s formula unchanged (the GNN/PointNet module classes share the house-points kwargs shape).
- [x] 1.7 Override it on `VGGTHouseGlobalEmbeddingEncoder` to emit `embed_dim`/`token_dim`/`num_patch_tokens`/`reducer_hidden`/`reducer_layers`/`camera_hidden`/`camera_layers`. No `rgb_branch` (stripped by the rich-API restoration, 9d919fb) and no `compute_dtype`.
- [x] 1.8 `VGGTHouseContextEncoder` and its `nogate` subclasses do NOT override the method — they inherit `VGGTEncoder`'s dispatch (the `HybridEncoder` branch covers `house_context`; the `TokenTransformerEncoder` branch covers the no-gate house-token encoders).

## 2. Make `encoder_module_kwargs_from_config` a delegating shim

- [x] 2.1 Replace the per-class table body of `encoder_module_kwargs_from_config` in `src/r2dreamer/observation_preparation/contracts.py` with a delegation to `encoder_registry[config.encoder_type].module_kwargs_from_config(config)` (lazy import of the registry to avoid cycles).
- [x] 2.2 Keep the `(config, encoder_module_cls)` signature; `encoder_module_cls` is documented as deprecated/ignored in the docstring.
- [x] 2.3 Remove `_kwargs_dispatch_name`, `_tuple_value`, `_HANDLED_ENCODER_CLASS_NAMES`, and the `if class_name == ...` kwargs ladder from `contracts.py`.

## 3. Collapse `factory.py` kwargs builders

- [x] 3.1 Deviation: `_contract_encoder_kwargs` is KEPT in `src/r2dreamer/encoders/factory.py`. The snapshot-first construction path (build from `cfg.encoder_input_contract["encoder_module_kwargs"]` when a contract snapshot is present) is required for eval-from-checkpoint fidelity and for `test_agent_instantiates_encoder_module_from_contract_kwargs`; removing it would break both. This matches the design Non-Goal "eval-from-checkpoint uses the stored snapshot."
- [x] 3.2 Each `_make_<encoder>` body now delegates to `encoder_module_kwargs_from_config(cfg, cls)` for the fresh-build path, preserving only the per-encoder `compute_dtype` overlay. The config→kwargs mapping bodies are gone.
- [x] 3.3 The `_make_house_global_embedding_encoder` `num_patch_tokens` computation now lives on `VGGTHouseGlobalEmbeddingEncoder.module_kwargs_from_config` (step 1.7), not in the factory.
- [x] 3.4 `compute_dtype` stays a factory-only overlay (NOT emitted by `module_kwargs_from_config`): it is a JAX dtype, not JSON-serializable, so it cannot enter the durable contract snapshot. The factory's per-encoder dtype handling (full_bf16-gated for conv/house_points; always-on for rgb_token/token_transformer; none for the rest) is preserved exactly in the `_make_*` overlays.
- [x] 3.5 `house_point_norm` is folded INTO `VGGTHousePointsPoseEncoder.module_kwargs_from_config` (additive to the snapshot, fixes a latent eval desync for `vggt_hybrid_house_points_pose` with a non-default norm); the factory's hand-built `house_point_norm` and `HybridHousePointsCameraEncoder` `cnn_*` block are removed (the launcher kwargs own them now).

## 4. Verify call sites are unchanged

- [x] 4.1 `launch/train.py` still calls `encoder_module_kwargs_from_config(config, ...)` and now gets launcher-derived kwargs via the shim.
- [x] 4.2 `checkpointing.py` still calls the shim and writes the launcher-derived `encoder_module_kwargs` into the snapshot.
- [x] 4.3 `launch/evaluate.py` contract-recovery path is unchanged (uses the stored `encoder_module_kwargs` snapshot via `_contract_encoder_kwargs`).
- [x] 4.4 Added `VGGTAggRawEncoder` (bare variant via `variant_encoder_class("VGGTAggRawEncoder", "vggt_agg_raw")`) and registered it in `encoder_registry`. `vggt_agg_raw` was present in `factory._resolve_encoder_cls` and `VGGT_DREAMER_SPECS` but missing from `encoder_registry`, so the shim's registry delegation would have `KeyError`'d for it. The registry now covers all 18 encoder types.

## 5. Tests

- [x] 5.1 `tests/r2dreamer/launch/test_registries.py::test_module_constructs_from_contract_kwargs` passes for all 18 `encoder_registry` keys (parametrized over `sorted(encoder_registry)`), each constructing its `module_cls` via the shim.
- [x] 5.2 Added `test_factory_and_resolver_agree_on_kwargs` (parametrized over `vggt` and `vggt_house_global_embedding`): asserts `config.full_bf16 is False` and that the factory fresh-build module's attributes equal the resolver kwargs for every key — the contract-snapshot path and the agent-factory path return equal kwargs on no-dtype encoders.
- [x] 5.3 `openspec validate --strict` on this change (run after these artifact edits).
- [x] 5.4 Full `tests/r2dreamer/` suite: 243 passed, 11 failed, 3 skipped. All 11 failures are pre-existing environment errors unrelated to this change (10 `habitat-*` preset/curriculum tests failing on `Curriculum file missing: data/curriculum/...`, plus `TestHabitatCurricula::test_all_curriculum_paths_exist`); the 2 pre-existing collection-error files (`test_encoders.py` stale import, `test_cross_framework.py` missing `omegaconf`) were excluded. Zero regressions from the consolidation; the new regression test passes.