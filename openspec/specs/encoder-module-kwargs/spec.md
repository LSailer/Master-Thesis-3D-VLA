# encoder-module-kwargs Specification

## Purpose
TBD - created by archiving change consolidate-encoder-module-kwargs. Update Purpose after archive.
## Requirements
### Requirement: Launcher Encoder owns its module constructor kwargs

Each launcher `Encoder` subclass SHALL provide a `module_kwargs_from_config(cls, config)`
classmethod returning the `dict` of constructor kwargs for its `module_cls` Flax module,
derived from the effective agent config. This method SHALL be the single source of truth for
that encoder's config→module-kwargs mapping.

#### Scenario: Constructing the module for any registry encoder
- **WHEN** `encoder_registry[encoder_type].module_kwargs_from_config(config)` is called with a valid config
- **THEN** the returned dict SHALL be accepted by `module_cls(**kwargs)` without a `TypeError`

#### Scenario: Signature change stays in sync
- **WHEN** a launcher encoder's `module_cls` constructor signature changes
- **THEN** only that encoder's `module_kwargs_from_config` SHALL need updating, and no other encoder's kwargs resolution SHALL be affected

#### Scenario: Variant-driven encoders share one implementation
- **WHEN** the encoder is a `VGGTEncoder` subclass that sets only `variant_key`
- **THEN** `module_kwargs_from_config` SHALL resolve from the variant spec without the subclass overriding the method

### Requirement: Contract snapshot delegates to the launcher Encoder

`encoder_module_kwargs_from_config(config, encoder_module_cls)` SHALL resolve kwargs by
delegating to `encoder_registry[config.encoder_type].module_kwargs_from_config(config)`.
It SHALL NOT maintain its own per-class kwargs table or dispatch by module-class name.

#### Scenario: Contract snapshot write path
- **WHEN** `encoder_input_contract_snapshot` builds the contract snapshot at checkpoint write time
- **THEN** `encoder_module_kwargs["encoder_module_kwargs"]` SHALL equal
  `encoder_registry[config.encoder_type].module_kwargs_from_config(config)`

#### Scenario: Public signature preserved
- **WHEN** a caller invokes `encoder_module_kwargs_from_config(config, encoder_module_cls)`
- **THEN** the function SHALL accept both arguments as before (the `encoder_module_cls` argument MAY be unused) and return the kwargs dict

### Requirement: Agent module factory delegates to the launcher Encoder

The agent-side module factory (`src/r2dreamer/encoders/factory.py`) SHALL build each Flax
encoder module by calling its launcher encoder's `module_kwargs_from_config(config)`, not
from a separate per-class kwargs builder. The factory SHALL NOT duplicate the
config→kwargs mapping.

#### Scenario: Agent builds module from config
- **WHEN** the agent constructs its encoder module for a given `encoder_type`
- **THEN** the module SHALL be constructed as `module_cls(**encoder_registry[encoder_type].module_kwargs_from_config(config))`

#### Scenario: No duplicate kwargs logic
- **WHEN** the factory resolves kwargs for any encoder
- **THEN** it SHALL obtain them solely from `module_kwargs_from_config`, and no `_make_*` function in `factory.py` SHALL contain a config→kwargs mapping body

### Requirement: Single resolver path keeps encoders constructible

The contract-snapshot path and the agent-factory path SHALL resolve module kwargs through the
same `module_kwargs_from_config` classmethod, so that a drift between the two is structurally
impossible.

#### Scenario: Regression test passes for every registry encoder
- **WHEN** `test_module_constructs_from_contract_kwargs` runs across every key in `encoder_registry`
- **THEN** each `module_cls(**encoder_module_kwargs_from_config(config, module_cls))` SHALL succeed and return an instance of `module_cls`

