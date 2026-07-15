## ADDED Requirements

### Requirement: A launcher `Encoder` selection owns its variant facts
Each launcher `Encoder` subclass SHALL declare its own `readout`
(`HeadReadout` | `TokenReadout`), `storage` (`StorageSpec`), `module_cls`, and
`module_kwargs_from_config`, and SHALL derive `feature_kind`, `compute_heads`,
`wp_pool_size`, and `encoder_type` from those declared fields. No separate
keyed table SHALL hold this data.

#### Scenario: A variant's facts are readable from its class alone
- **WHEN** a caller holds an `Encoder` subclass
- **THEN** its readout, storage, module class, and derived `feature_kind` /
  `compute_heads` / `wp_pool_size` are reachable without a lookup through a
  string key or a variant table

#### Scenario: Derived properties keep their existing rules
- **WHEN** `compute_heads` is read on any selection
- **THEN** it equals `isinstance(readout, HeadReadout)` — the VGGT point/camera
  heads run for head-readout variants and are skipped for token-readout
  variants, unchanged from the retired `VGGTDreamerSpec`

### Requirement: Each variant identifier is declared once in `src/`
A variant's identifier SHALL appear exactly once under `src/`: on its `Encoder`
subclass. `encoder_registry` SHALL derive its key from the class rather than
restating it, and no value SHALL restate the key that addresses it.

#### Scenario: Registry key is derived, not typed
- **WHEN** an encoder selection is registered
- **THEN** its registry key is read from the class attribute, and adding a
  variant requires typing the identifier in one place

#### Scenario: A dropped registration fails loudly
- **WHEN** an encoder module is not imported and so never registers
- **THEN** the registry key-set test fails at test time, rather than surfacing
  as an "unknown encoder" `KeyError` at launch

### Requirement: Registration is import-time and explicitly triggered
Encoder registration SHALL occur as an import side effect. `registries.py`
SHALL retain an explicit import list as the discovery mechanism. Automatic
module discovery SHALL NOT be used.

#### Scenario: Synthesised subclasses are registered
- **WHEN** an `Encoder` subclass is created dynamically rather than at a `def`
  site
- **THEN** it is registered at its construction site, and the registry key-set
  test covers it identically to hand-written classes

### Requirement: `encoders/` does not depend on `observation_preparation/`
No module under `src/r2dreamer/encoders/` SHALL import
`src/r2dreamer/observation_preparation/`. The dependency SHALL point one way:
`observation_preparation/` → `encoders/`.

#### Scenario: The lazy-import workaround is gone
- **WHEN** `src.r2dreamer.encoders` is imported
- **THEN** no lazy mapping or descriptor defers an
  `observation_preparation` import, because no such import exists

## MODIFIED Requirements

### Requirement: Durable contract snapshots are unchanged by this refactor
`EncoderInputContract.to_snapshot()` SHALL produce byte-identical output for
every registered encoder before and after this change, at `version: 1`. No Flax
`nn.Module` class SHALL change its `__module__` or `__qualname__`, because
snapshots persist `module_class_path(encoder_module_cls)` and recover it via
`_import_class`.

#### Scenario: Pre-change checkpoints still load
- **WHEN** a checkpoint written before this change is loaded by `evaluate` or
  `R2DreamerAgent.from_checkpoint`
- **THEN** `EncoderInputContract.from_snapshot` resolves `encoder_module` to the
  same class, and the encoder input shape and type match, exactly as before

#### Scenario: Snapshot equality is enforced by fixture
- **WHEN** the golden contract fixture is regenerated after the refactor
- **THEN** it is byte-identical to the fixture captured on `main`, for all 18
  registry entries
