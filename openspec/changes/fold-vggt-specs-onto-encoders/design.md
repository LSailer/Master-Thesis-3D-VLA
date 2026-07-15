## Context

`VGGTDreamerSpec` (`src/r2dreamer/observation_preparation/vggt.py`) is a good
abstraction in the wrong package. It already collapses the variant's axes into
one record — `readout`, `storage`, `dreamer`, `agent_overrides`, `design_notes` —
and derives `feature_kind`, `compute_heads`, `wp_pool_size` from them. The
problem is only that the record sits a package away from the `Encoder` class it
describes, reachable through a string key.

Today's resolution path for `VGGTHouseContextEncoder`:

```
encoder_registry["vggt_house_context"]   registries.py     (string #1)
        │
        ▼
VGGTHouseContextEncoder.variant_key = "vggt_house_context"  (string #2)
        │
        ▼  _VariantDescriptor.__get__
VGGT_VARIANTS  ──▶ _LazyVGGTVariants._data()
        │              └─ import_module("...observation_preparation.vggt")
        ▼
VGGT_DREAMER_SPECS["vggt_house_context"]                    (string #3)
        │
        ▼
VGGTDreamerSpec(name="vggt_house_context", ...)             (string #4)
```

Target:

```
encoder_registry ◀── @register  (key read from the class)
        │
        ▼
class VGGTHouseContextEncoder(VGGTEncoder):
    variant_key = "vggt_house_context"            (the only occurrence)
    readout  = TokenReadout(token_source="full", ...)
    storage  = StorageSpec(replay_rgb=True, replay_readout=True)
    module_cls = TokenTransformerEncoder          (direct import, same package)
```

## Goals / Non-Goals

**Goals**
- One occurrence of each variant identifier in `src/`.
- `feature_kind` / `compute_heads` / `wp_pool_size` stay *derived*, never stored
  twice.
- Delete `_LazyVGGTVariants`, `_VariantDescriptor`, `VGGT_VARIANTS`.
- Collapse both `encoder_type ==` if-chains into per-class overrides.

**Non-Goals**
- Moving any `nn.Module`. See the checkpoint constraint below.
- Changing shapes, numerics, defaults, or CLI surface.
- Auto-discovery of encoder modules (`pkgutil.walk_packages`). Registration is
  import-time; an explicit import list stays.

## Key Decision 1 — checkpoint durability is the hard boundary

`EncoderInputContract.to_snapshot()` writes:

```python
"encoder_module": module_class_path(self.encoder_module_cls)
#  -> "src.r2dreamer.encoders.transformer.TokenTransformerEncoder"
```

and `from_snapshot()` reverses it with `_import_class(...)`. The string is
`f"{cls.__module__}.{cls.__qualname__}"`. Any `nn.Module` that changes file or
name silently invalidates every checkpoint that references it — discovered only
when `evaluate` tries to load.

This partitions the codebase for this change:

| Layer | Files | May move? |
| --- | --- | --- |
| Flax `nn.Module` classes | `encoders/mlp.py`, `transformer.py`, `cnn.py`, `pointnet.py`, `gnn_house.py` | **No** |
| Launcher `Encoder` selections | `encoders/base.py`, `encoders/__init__.py`, `house_points_pose.py`, … | Yes |
| Spec data | `observation_preparation/vggt.py` | Yes (this is what moves) |

The fold only touches the bottom two rows.

**Acceptance test.** Before any edit, capture `spec().contract_snapshot` (or
`contract.to_snapshot()`) for all 18 registry entries to a golden JSON fixture.
After the refactor, assert byte-equality. This is not a proxy for correctness —
it *is* the contract that `evaluate` and `from_checkpoint` depend on.

## Key Decision 2 — the fold removes the import cycle

The lazy mapping is not incidental; it is load-bearing today:

```
encoders/base.py ──lazy──▶ observation_preparation/vggt.py
        ▲                            │
        └────────────────────────────┘
             imports ConvEncoder, MLPEncoder,
             TokenTransformerEncoder, constants
```

Verified: `encoders/mlp.py`, `transformer.py`, `cnn.py`, `constants.py` import
nothing from `base.py` or `observation_preparation`. So once the spec lives on
the encoder class, `base.py` can import the Flax modules directly and the arrow
only points one way:

```
observation_preparation/vggt.py ──▶ encoders/base.py ──▶ encoders/{mlp,transformer,cnn}.py
```

`_LazyVGGTVariants` and `_VariantDescriptor` then have nothing to do. This is
the main structural argument for the change: it *deletes* a workaround.

**Risk:** `base.py` gains eager imports of `mlp.py` / `transformer.py`, which
pull `world_model/heads.py` and `world_model/rssm.py` (both import flax/jax).
`base.py` already imports `encoders/cnn.py`, which already imports
`world_model/rssm.py` — so flax/jax is already on the `base.py` import path and
the eager cost is bounded. Confirm import time does not regress; if it does,
keep `module_cls` behind a per-class `classmethod` rather than a class
attribute.

**Measured baseline (task 1.5).** `python -X importtime -c "import
src.r2dreamer.encoders"`, cumulative for `src.r2dreamer.encoders`, five warm
runs on the mac: 554 / 436 / 321 / 330 / 382 ms — median ~382 ms, spread ~±30%
run to run. Task 2.8 should treat anything inside ~600 ms as noise and compare
medians, not single runs.

The baseline also **retires most of this risk**. Of the ~331 ms in a
representative run, `flax.linen` accounts for ~265 ms (~80%), reached through
`encoders.base` → `encoders.cnn` → `world_model/rssm.py`. And after `import
src.r2dreamer.encoders`, `sys.modules` already contains
`src.r2dreamer.encoders.mlp` and `src.r2dreamer.encoders.transformer` — the
package `__init__.py` imports `TokenTransformerEncoder` directly, and `mlp.py`
arrives transitively. So the eager imports Decision 2 worries about load **no
module that is not already loaded**; only the import order inside the package
changes. `observation_preparation` is correctly absent from `sys.modules` —
that, and only that, is what the lazy mapping defers, and it is what the fold
removes.

Expect 2.8 to be a formality. The `classmethod module_cls` fallback should not
be needed; if 2.8 shows a median regression anyway, suspect a new import edge
rather than the mlp/transformer ones.

## Key Decision 3 — decorator registration, scoped honestly

Registration removes exactly one duplication: the `registries.py` dict key.
Worth doing *as part of* this change (the class already carries the key), not
worth doing alone.

```python
@register_encoder                    # key = cls.variant_key / cls.encoder_type
class VGGTHouseContextEncoder(VGGTEncoder):
    variant_key = "vggt_house_context"
```

Constraints that shape the design:

- **Import-time, not discovery-time.** A decorator only runs if the module is
  imported. `registries.py` keeps its explicit import list; the dict literal
  becomes an import list. This is a real trade — the 18-entry table is no longer
  visible on one screen. Mitigate with a test asserting the registry's exact key
  set, so a dropped import fails loudly rather than surfacing as "unknown
  encoder" at launch.
- **Six encoders have no `def` site.** `VGGTAggregatorMLPEncoder` and friends are
  built by `variant_encoder_class(name, key)` via `type(...)`. That factory
  registers its product directly; the decorator covers the hand-written classes.
  Two registration paths, one registry — acceptable, but the key-set test is
  what keeps them honest.
- **`encoder_type` is a property, not a class attribute,** on `VGGTEncoder` (it
  reads through `self.variant`). The decorator must key off `variant_key` for
  VGGT variants and `encoder_type` for `CNNEncoder`. Post-fold, prefer a single
  uniform class-level `variant_key` on every selection to avoid a two-branch
  decorator.

## Risks

| Risk | Mitigation |
| --- | --- |
| Silent checkpoint breakage | Golden `to_snapshot()` fixture, all 18 encoders, byte-equality |
| Dropped import ⇒ missing encoder | Test asserting exact registry key set |
| Import-time regression from eager Flax imports | Measure; fall back to `classmethod` `module_cls` |
| Behaviour drift while collapsing if-chains | Chains are pure `config → kwargs`; assert resolved kwargs per encoder before/after |
| `encoders/pointnet2.py` is a skeleton for the external TF1 `pointnet2` repo and defines no launcher `Encoder` (its `encoder_type = "pointnet2"` is a `NamedTuple` field default on `PointNet2PipelineSpec`). Correctly absent from `encoder_registry` | Leave untouched; the key-set test pins the 18 real entries and its absence with it |

## Key Decision 4 — `StorageSpec` lives on the `Encoder`

**Decided: `storage` is a declared field on the `Encoder` subclass**, alongside
`readout` and `module_cls`.

The objection was that `StorageSpec` describes replay, not launch, so it looked
like it belonged in `observation_preparation/`.

**`storage` is an independent axis — it is NOT derivable from `readout`.** The
current table proves it, with identical readouts taking opposite storage:

| readout | variant | `replay_rgb` |
| --- | --- | --- |
| `HeadReadout(37)` | `vggt` | `False` |
| `HeadReadout(37)` | `hybrid` | **`True`** |
| `TokenReadout("global")` | `vggt_agg_token_transformer` | `False` |
| `TokenReadout("global")` | `vggt_house_global_tokens_nogate` | **`True`** |

So `storage` must stay a *declared* field. Any attempt to derive it from
`readout` encodes a coincidence that the table already contradicts. This also
kills the tempting "one readout ⇒ one storage" simplification.

The reason it belongs on the `Encoder` is therefore not derivation but
ownership: `storage` is **per-variant data, and the variant is the `Encoder`
class**. Same argument as `readout` and `module_cls` — a fact about one variant
belongs on the object that is that variant.

The "replay is not launch" objection does not survive contact with what the
selection already carries. `agent_overrides` holds `buffer_capacity`
(`{"buffer_capacity": 1_000_000}` on the head-readout variants) and
`_SMALL_REPLAY_OVERRIDES` holds `batch_size` / `seq_len` / `train_ratio`. Replay
concerns already live on the selection. `storage` joins existing company rather
than setting a precedent.

The alternative — leaving `storage` in `observation_preparation/` keyed off the
encoder class instead of a string — removes the string but keeps two facts about
one variant in two packages, and keeps a lookup on the read path. It buys
nothing the fold does not already buy.

Consequences:

- `StorageSpec` moves to a module importable by both packages without a cycle
  (task 2.2). `observation_preparation/` reads `EncoderCls.storage`; the arrow
  stays one-way.
- `tests/r2dreamer/test_observation_preparation.py` currently asserts
  `VGGT_DREAMER_SPECS["vggt"].storage.replay_rgb is False`. These become
  assertions on the encoder class — same meaning, new subject (task 5.2).
- Add a regression test pinning the four rows above, so a future "simplify" pass
  cannot quietly derive `storage` from `readout`.

## Open Questions

- `tests/r2dreamer/launch/test_encoders.py` asserts identity
  (`VGGTEncoder.variant is VGGT_VARIANTS["vggt"]`). Post-fold `variant` has no
  referent — do these tests get rewritten against `readout`/`storage`, or is a
  compatibility `variant` property worth keeping?
