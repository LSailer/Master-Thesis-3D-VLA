## Context

Today, deriving a Flax module's constructor kwargs from an `R2DreamerConfig` is done in
**two** places that must agree by hand:

1. `src/r2dreamer/observation_preparation/contracts.py` —
   `encoder_module_kwargs_from_config(config, encoder_module_cls)` is a per-class table
   dispatching on `_kwargs_dispatch_name(encoder_module_cls)` (a "nearest handled ancestor"
   walk up the class hierarchy). This is the **contract-snapshot path**, called from
   `checkpointing.py:84` (write) and `launch/train.py:202` (fresh build).
2. `src/r2dreamer/encoders/factory.py` — `_contract_encoder_kwargs(cfg)` plus the
   `_make_<encoder>` functions. This is the **agent.py path**, called when the agent
   builds its module from a config.

A `test_module_constructs_from_contract_kwargs` regression test exists solely to keep the
two in sync. It is failing for `vggt_house_global_embedding` (Cause A): the `contracts.py`
table emits `embed_dim`/`token_dim`/`num_patch_tokens`/`reducer_hidden`/`reducer_layers`/
`camera_hidden`/`camera_layers`/`rgb_branch`, but `HouseGlobalEmbeddingEncoder.__init__`
drifted to `(mlp_layers, hidden_dim)` — so construction raises
`unexpected keyword argument 'embed_dim'`.

Verified at all three resolver call sites — `launch/train.py:202`, `checkpointing.py:84`,
and `factory.py` `_make_*` — the launcher `Encoder` that owns this module is **already in
scope** (train has `encoder_spec` from `encoder_registry[encoder].from_train_args(args).spec()`;
checkpointing has `config.encoder_type`; factory has `cfg.encoder_module_cls` set by the
launcher). The module-class-name dispatch in the resolver exists by convention, not
necessity — it was written to avoid depending on the launcher.

The launcher `Encoder` already exposes `module_cls` (the Flax class) and `agent_overrides`
through the variant spec; it is the natural home for "what kwargs does my module take."

## Goals / Non-Goals

**Goals:**
- One source of truth per encoder for config→module-constructor kwargs, co-located with
  `module_cls`, so a module signature change cannot outrun a faraway table.
- Delete the duplicate kwargs table (`contracts.py`) and the duplicate kwargs bodies
  (`factory.py` `_make_*`).
- Keep `encoder_module_kwargs_from_config(config, module_cls)` as a stable public entry
  point with its current signature — callers (train/eval/checkpoint) are unchanged.

**Non-Goals:**
- Restoring the `HouseGlobalEmbeddingEncoder` rich API (separate pending decision). This
  change assumes each module's `__init__` already accepts the kwargs its encoder emits.
- Fixing the stale `encode_house_global_obs` import in `tests/r2dreamer/launch/test_encoders.py`.
- Any environment-error fix (habitat/omegaconf/ruamel/curriculum-data/pose-buffer).
- Changing checkpoint on-disk format or the contract-snapshot field names
  (`encoder_module_kwargs`, `encoder_module_cls`).

## Decisions

### D1: Put `module_kwargs_from_config` on the launcher `Encoder`, not on the Flax module

**Choice:** classmethod `module_kwargs_from_config(cls, config) -> dict` on
`src/r2dreamer/encoders/base.py:Encoder`, overridden per launcher subclass.

**Why over the alternatives:**
- *On the Flax `nn.Module` class itself?* Rejected: the Flax module would then import
  `R2DreamerConfig` and know config field names (`vggt_embed_dim`, `mlp_vggt_hidden`, …),
  coupling the pure trainable module to the agent config schema. The launcher is already
  config-aware and adapter/extractor-coupled; the Flax module is deliberately not.
- *Keep the central `contracts.py` table but auto-generate it from class signatures?*
  Rejected: introspecting `nn.Module` dataclass fields can't express the config-field
  *mapping* (which config field feeds which kwarg, plus computed values like
  `num_patch_tokens = vggt_token_count - (1 + AGG_REGISTER_TOKENS)`). The mapping is real
  logic, not reflection.

### D2: `VGGTEncoder` base implements it once for variant-driven encoders

The `variant_encoder_class(...)`-generated encoders and `VGGTEncoder` subclasses that only
set `variant_key` get `module_kwargs_from_config` from `VGGTEncoder`. **Implemented form
(deviation from the original D2 wording):** `VGGTDreamerSpec` carries the module *class*
(`module_cls`) but no kwargs *values*, so reading "kwargs from the variant spec
(`self.variant`)" is infeasible for values. Instead `VGGTEncoder.module_kwargs_from_config`
resolves the module's **identity** from `VGGT_VARIANTS[cls.variant_key].module_cls` and reads
the **values** from `config` via a module-name dispatch helper `_vggt_module_kwargs(module_cls,
config)` co-located in `base.py`. That helper is the single source of truth for every
VGGT-variant module's config→kwargs formula (ConvEncoder / WP64CNNCPMLPEncoder / HybridEncoder /
TokenTransformerEncoder branches + an MLP-tail fallthrough), mirroring the former
`contracts.py` table verbatim so durable snapshots are unchanged.

Standalone encoders that already override `module_cls` as a property
(`VGGTHouseGlobalEmbeddingEncoder`, `VGGTHousePointsPoseEncoder`,
`VGGTHybridHousePointsPoseEncoder`) override the classmethod themselves. `HybridEncoder`,
`VGGTHouseContextEncoder`, the no-gate house-token encoders, and the GNN/PointNet
house-points variants do NOT override it — they inherit `VGGTEncoder`'s dispatch (their
modules land on an existing `_vggt_module_kwargs` branch) or `VGGTHousePointsPoseEncoder`'s
formula, respectively. `CNNEncoder` overrides it with the plain CNN kwargs. This mirrors the
existing split: the variant spec identifies the module; the launcher subclass owns the
config→kwargs formula where a subclass already exists.

### D3: `encoder_module_kwargs_from_config` becomes a one-line registry-delegating shim

```python
def encoder_module_kwargs_from_config(config, encoder_module_cls):
    from src.r2dreamer.encoders import encoder_registry  # lazy
    return encoder_registry[config.encoder_type].module_kwargs_from_config(config)
```

**Why keep the shim** instead of having callers do the registry lookup directly:
- Preserves the existing `(config, encoder_module_cls)` signature at all call sites —
  train/eval/checkpoint don't change.
- `encoder_module_cls` becomes an unused parameter (kept for signature compatibility;
  the dispatcher no longer needs it). This is the one mild wart; acceptable to avoid
  churning three call sites and to keep the contract-snapshot's stored signature stable.

**Why not drop the `module_cls` arg:** it is part of the recorded contract snapshot and the
public function signature; dropping it is a broader API change than this consolidation
warrants. Flagged for a future cleanup.

### D4: `factory.py` `_make_*` collapse to the classmethod

Each `_make_<encoder>(cfg, cls)` shrinks to delegate to
`encoder_module_kwargs_from_config(cfg, cls)` for the fresh-build path, where it still owns
`cls` resolution. The config→kwargs mapping bodies are gone. **Deviation:** the
`_contract_encoder_kwargs` helper is KEPT — it implements the snapshot-first construction
path (build from `cfg.encoder_input_contract["encoder_module_kwargs"]` when a contract
snapshot is present), which eval-from-checkpoint and
`test_agent_instantiates_encoder_module_from_contract_kwargs` depend on; removing it would
break both and would contradict the Non-Goal that eval uses the stored snapshot. The
factory's per-encoder `compute_dtype` overlays are also kept (see D5). The agent.py fresh-build
path thus reads the same single source of truth as the contract-snapshot write path.

### D5: `compute_dtype` and `house_point_norm` placement (resolved in implementation)

- **`compute_dtype` stays a factory-only overlay.** It is a JAX dtype, not JSON-serializable,
  so it cannot enter the durable contract snapshot, and therefore cannot be emitted by
  `module_kwargs_from_config` (which feeds the snapshot). The factory's inconsistent
  per-encoder dtype handling (full_bf16-gated for conv/house_points; always-on for
  rgb_token/token_transformer; none for wp_conv/wp64/hybrid/house_global/mlp) is preserved
  exactly in the `_make_*` overlays.
- **`house_point_norm` is folded INTO `VGGTHousePointsPoseEncoder.module_kwargs_from_config`.**
  The old `contracts.py` table omitted it while `factory.py` passed it — a latent desync that
  would silently mis-evaluate `vggt_hybrid_house_points_pose` from a checkpoint with a
  non-default norm. Folding it into the launcher kwargs (an additive snapshot change) makes
  the snapshot authoritative and closes that desync.

## Risks / Trade-offs

- **[Re-couples fresh-build to the registry key]** `encoder_registry[config.encoder_type]`
  now drives kwargs. If a registry key is renamed, a config from an old run can't fresh-build.
  → **Mitigation**: eval-from-checkpoint is unaffected (it uses the stored
  `encoder_module_kwargs` snapshot, not the resolver). Only fresh-build/train from an old
  config string is exposed — same exposure the registry already has for *selection*. Net
  fragility is not increased relative to selecting the encoder by that string in the first
  place.
- **[Config field-name knowledge spreads to ~17 encoder files]** Renaming a config field
  touches more files. → **Mitigation**: each touch is local and obvious, and a
  `module_cls(**wrong_kwargs)` fails loudly at construction (as Cause A showed) instead of
  silently desyncing two tables. The `test_module_constructs_from_contract_kwargs` test
  catches per-encoder drift immediately.
- **[Unused `encoder_module_cls` param on the shim]** Mild API wart. → **Mitigation**:
  documented as deprecated; kept for signature stability. A follow-up change can drop it
  across the three call sites.
- **[Variant-spec kwargs coverage]** `VGGTEncoder.module_kwargs_from_config` uses the variant
  spec only for module *identity* and reads all *values* from `config` via the
  `_vggt_module_kwargs` dispatch (see revised D2). The variant spec carries no kwargs values,
  so coverage is determined by which `_vggt_module_kwargs` branch the resolved `module_cls`
  lands on. → **Mitigation**: the dispatch branches mirror the former `contracts.py` table
  verbatim, so coverage is unchanged; where the contract table computed extra kwargs (e.g.
  `num_patch_tokens`), that logic moved into the relevant encoder's
  `module_kwargs_from_config` (the standalone ones). The
  `test_module_constructs_from_contract_kwargs` regression (now parametrized over all 18
  registry keys) verifies each emitted kwargs set is accepted by its `module_cls`. This change
  assumes each encoder's emitted kwargs already match its module signature (the rich-API
  restoration was the prerequisite, landed in 9d919fb).
- **[Registry coverage]** The shim delegates via `encoder_registry[config.encoder_type]`, so
  every `encoder_type` the factory and `VGGT_DREAMER_SPECS` know must also be registered.
  `vggt_agg_raw` was missing from `encoder_registry` and would have `KeyError`'d. →
  **Mitigation**: added the `VGGTAggRawEncoder` bare variant + registry entry; the registry
  now covers all 18 encoder types, verified by the parametrized regression test.