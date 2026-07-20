## Why

One VGGT variant is described in four places that must agree by hand.

Take `vggt_house_global_tokens_nogate`. Its identifier appears 13 times across
10 files. Four of those are the *same fact* restated:

1. `encoder_registry["vggt_house_global_tokens_nogate"]` in
   `src/r2dreamer/launch/registries.py` — the launch key.
2. `VGGTHouseGlobalTokenNoGateEncoder.variant_key = "vggt_house_global_tokens_nogate"`
   in `src/r2dreamer/encoders/__init__.py` — the same string, on the class the
   registry points at.
3. `VGGT_DREAMER_SPECS["vggt_house_global_tokens_nogate"]` in
   `src/r2dreamer/observation_preparation/vggt.py` — keyed by the same string.
4. `VGGTDreamerSpec(name="vggt_house_global_tokens_nogate", ...)` — the dict key
   restated *inside* the value it keys.

The registry key is derivable from `variant_key` (true for all 18 entries). The
spec dict key is derivable from `name` (true for all 12 entries). Neither
derivation is made; both are typed twice. `launch_run` in
`scripts/r2dreamer/_run_configs.py` validates `cfg["encoder"]` against
`encoder_registry` precisely because this class of typo is reachable.

The same strings then drive **two parallel if-chains**: 7 `encoder_type ==`
branches in `_vggt_module_kwargs` (`src/r2dreamer/encoders/base.py`) and 7 more
in `src/r2dreamer/encoders/factory.py`. Adding a variant means touching a dict,
a class attribute, a spec table, and up to two if-chains — with nothing that
fails loudly if one is missed.

This change finishes a direction already started. The previous commit
("consolidate encoder-module kwargs onto launcher Encoder") moved the
config→kwargs formula next to `module_cls` so the constructor signature and the
resolved kwargs "cannot desync (the structural fix for the Cause A drift)". The
readout, storage, and module identity that `VGGTDreamerSpec` holds are the same
kind of fact and still live a package away.

Folding the spec onto the class also **removes** an existing workaround rather
than fighting it. `observation_preparation/vggt.py` imports `ConvEncoder`,
`MLPEncoder`, and `TokenTransformerEncoder` from `src/r2dreamer/encoders/`,
while `encoders/base.py` needs the specs — a cycle, broken today by
`_LazyVGGTVariants` ("Lazy mapping to avoid importing Observation Preparation
during package init") and the `_VariantDescriptor` that reads through it. The
Flax module files (`mlp.py`, `transformer.py`, `cnn.py`, `constants.py`) import
nothing from `base.py` or `observation_preparation`, so a spec living on the
encoder class can name its `module_cls` directly. The cycle disappears, and
with it the lazy mapping and the descriptor.

## What Changes

- Each launcher `Encoder` subclass owns its variant facts directly: `readout`
  (`HeadReadout` | `TokenReadout`), `storage` (`StorageSpec`), `module_cls`, and
  the existing `module_kwargs_from_config` — replacing the `variant_key` →
  `VGGT_VARIANTS` → `VGGTDreamerSpec` indirection.
- `VGGT_DREAMER_SPECS` and the `VGGTDreamerSpec` dataclass are retired. Their
  derived properties (`feature_kind`, `compute_heads`, `wp_pool_size`,
  `encoder_type`) become methods/properties on the encoder class, computed from
  the same `readout`/`storage` fields by the same rules.
- `_LazyVGGTVariants`, `VGGT_VARIANTS`, and `_VariantDescriptor` are deleted
  from `encoders/base.py`; `encoders/base.py` imports the Flax module classes
  directly.
- `encoder_registry` is populated by a registration decorator that reads the key
  off the class, so the key is stated once. `variant_encoder_class` registers
  the subclasses it synthesises. `registries.py` keeps an explicit import list
  as the discovery mechanism (registration is import-time; nothing is
  auto-discovered).
- The two `encoder_type ==` if-chains in `_vggt_module_kwargs` and `factory.py`
  collapse into per-class overrides.
- The reverse `feature_kind` → spec lookups inside `observation_preparation/vggt.py`
  (lines ~412-429, ~619) and `VGGT_DREAMER_SPECS.get(cfg.encoder_type)` in
  `observation_preparation/module_factory.py` resolve against the encoder
  classes instead.

## Non-Goals / Constraints

- **No `nn.Module` class may change module or qualname.** Checkpoints store
  `module_class_path(encoder_module_cls)` — a literal import path such as
  `"src.r2dreamer.encoders.transformer.TokenTransformerEncoder"` — via
  `EncoderInputContract.to_snapshot()`, and `from_snapshot()` calls
  `_import_class()` on it. Relocating a Flax module class breaks every existing
  checkpoint. Only launcher-side selection data moves.
- **`to_snapshot()` output must be byte-identical per encoder.** The snapshot is
  version-gated (`version: 1`, rejected otherwise) and is the acceptance test
  for this change. This mirrors the discipline already recorded in
  `_vggt_module_kwargs`: "Mirrors the former `contracts.py` table verbatim so
  durable snapshots are unchanged."
- No behaviour, numerics, shapes, or config knobs change. This is a
  data-location refactor.
- `RUN_CONFIGS` is out of scope. Its `output_dir`/`wandb_name`/`wandb_tags` are
  documented defaults (`train()`: "shim-supplied defaults — CLI flags from
  argparse override if provided"), relied on by
  `scripts/slurm/configs/profile_house_points_pose.yaml` and by bare
  `run.py <run-id>` calls. They are not dead.
