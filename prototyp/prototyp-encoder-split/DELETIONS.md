# DELETIONS — files removable after full migration

Deletions happen in migration steps 4–5 (HANDOFF.md), only after the golden
runs are green. Never delete anything in steps 1–3.

## Safe to delete

- `src/r2dreamer/encoders/factory.py` — cfg→module dispatch
  (`_resolve_encoder_cls`, all `_make_*` builders, `_dummy_encoder_obs`)
  replaced by RECIPES + inferred obs spec. Move `_make_rssm` into the
  world_model module first.
- `src/r2dreamer/encoder_types.py` — the type-name tuples
  (`FLAT_VGGT_ENCODER_TYPES`, `COMPOSITE_RGB_ENCODER_TYPES`,
  `RGB_BEARING_ENCODER_TYPES`, …) become per-recipe fields
  (e.g. `rgb_key: str | None`) instead of global name lists.
- Combo Flax classes in `src/r2dreamer/encoders/mlp.py`:
  `WMHybridEncoder`, `WP64CNNCPMLPEncoder`, `HybridHousePointsCameraEncoder`,
  `HouseGlobalEmbeddingEncoder` — reproduced declaratively by
  `CompositeSpec`. `MLPEncoder` itself stays (mechanism module).

## Mostly deletable (remnants move first)

- Launcher spec hierarchy: the `VGGTEncoder` subclasses in
  `src/r2dreamer/encoders/__init__.py` (`HybridEncoder`,
  `VGGTHouseContextEncoder`, `VGGTHouseFullTokenNoGateEncoder`,
  `VGGTHouseGlobalTokenNoGateEncoder`) and the `variant_encoder_class`
  wrappers. Their `_build_adapter_for_extractor` logic becomes the recipes'
  `make_adapter` functions; `agent_overrides` / `design_notes` move into the
  recipe as data. `src/r2dreamer/encoders/base.py` shrinks accordingly.
- `src/r2dreamer/agent.py` — dissolves into world_model / behavior / learner;
  the file disappears or becomes the thin learner.
- `src/r2dreamer/observation_preparation/contracts.py` and
  `module_factory.py` — the `encoder_module_kwargs_from_config` /
  `normalize_encoder_module_kwargs` config→kwargs machinery is obsolete once
  recipes construct modules explicitly. Keep the actual contract checks.
- `src/configs/agent_interface.py` — `obs_shape` is replaced by
  `latent_dim`; the file shrinks to two ints or merges into the learner
  config.

## Stays (for clarity)

- Mechanism modules: `encoders/cnn.py`, `pointnet.py`, `pointnet2.py`,
  `transformer.py`, `gnn_house.py`.
- All adapters, `experience.py`, buffer code, `src/shared/optim.py`,
  `world_model/`, `behavior/`.
- Variant-specific encoder tests: replaced by the parametrized recipe test,
  but only AFTER serving as parity references during migration — do not
  delete them earlier.
