# Tasks

Ordering matters: the golden fixture must exist and pass **before** any
production file is edited, or the acceptance test proves nothing.

## 1. Pin the contract (before touching src/)

- [ ] 1.1 Add `tests/r2dreamer/launch/test_encoder_contract_golden.py`: for every
      key in `encoder_registry`, build the selection via `from_train_args` with
      default train args and dump `spec().contract_snapshot` (falling back to
      `spec()` fields where `contract_snapshot` is `None`) to a committed JSON
      fixture.
- [ ] 1.2 Assert the fixture round-trips: `EncoderInputContract.from_snapshot()`
      on each entry resolves `encoder_module` via `_import_class` without error.
      This is the check that would catch a relocated `nn.Module`.
- [ ] 1.3 Add a registry key-set test: `set(encoder_registry) == {…18 literal
      keys…}`, and `set(env_registry) == {"habitat", "crafter"}`. This is what
      makes a dropped import fail loudly once the dict literal goes away.
- [ ] 1.4 Add a resolved-kwargs fixture: for each encoder, dump
      `module_kwargs_from_config(effective_config)` to JSON. Pins the two
      if-chains before they collapse.
- [ ] 1.5 Record baseline `python -X importtime -c "import
      src.r2dreamer.encoders"` total, for the Decision-2 import-time risk.
- [ ] 1.6 Verify 1.1–1.5 pass on `main` unmodified. Commit the fixtures.

## 2. Move the spec data onto the classes

- [ ] 2.1 Give `Encoder` class-level `readout` / `storage` attributes and derive
      `feature_kind`, `compute_heads`, `wp_pool_size`, `encoder_type` from them
      using the exact rules from `VGGTDreamerSpec` (`compute_heads =
      isinstance(readout, HeadReadout)`; the `token_source` → `feature_kind`
      mapping; `wp_side` handling incl. the `"dense"` case).
- [ ] 2.2 Move `HeadReadout`, `TokenReadout`, `StorageSpec` to a module importable
      by both `encoders/` and `observation_preparation/` without a cycle
      (`encoders/constants.py` imports nothing — a candidate). Do **not** move
      `EncoderInputContract`.
- [ ] 2.3 Port all 12 `VGGT_DREAMER_SPECS` entries onto their `Encoder`
      subclasses, one variant per commit where practical. Re-run 1.1/1.4 after
      each — a diff in the golden fixture means the port changed behaviour.
- [ ] 2.4 Replace `variant_encoder_class(name, key)` with subclasses carrying
      `readout`/`storage`/`module_cls` directly, or keep the factory and pass the
      fields instead of a key.
- [ ] 2.5 Delete `_LazyVGGTVariants`, `VGGT_VARIANTS`, `_VariantDescriptor`, and
      the `variant` property. Import Flax module classes directly in `base.py`.
- [ ] 2.6 Delete `VGGTDreamerSpec` and `VGGT_DREAMER_SPECS`. Repoint the reverse
      lookups in `observation_preparation/vggt.py` (~412-429, ~619) and
      `module_factory.py:75` at the encoder classes.
- [ ] 2.7 Re-check import direction: nothing under `encoders/` may import
      `observation_preparation`. Add an import-linter rule or a test asserting it.
- [ ] 2.8 Compare `-X importtime` against the 1.5 baseline. If materially worse,
      fall back to `classmethod module_cls` per Decision 2.

## 3. Collapse the if-chains

- [ ] 3.1 Move each `_vggt_module_kwargs` branch (7) onto the owning class as
      `module_kwargs_from_config`. Delete the function once empty.
- [ ] 3.2 Move each `factory.py` `encoder_type ==` branch (7) onto the owning
      class. Delete the chain.
- [ ] 3.3 Re-run 1.4. Resolved kwargs must be byte-identical.

## 4. Decorator registration

- [ ] 4.1 Add `@register_encoder` reading the key off the class; make the key
      attribute uniform across all selections first (see Decision 3 — today
      `CNNEncoder` uses `encoder_type`, VGGT variants use `variant_key`).
- [ ] 4.2 Decorate the hand-written classes; register synthesised ones at their
      construction site.
- [ ] 4.3 Replace the `registries.py` dict literal with the import list. Keep
      `env_registry` as a plain dict — two entries, no duplication to remove.
- [ ] 4.4 Re-run 1.3. The key set must be unchanged.

## 5. Close out

- [ ] 5.1 Rewrite `tests/r2dreamer/launch/test_encoders.py` identity assertions
      (`VGGTEncoder.variant is VGGT_VARIANTS["vggt"]`) against `readout`/`storage`.
- [ ] 5.2 Update `tests/r2dreamer/test_observation_preparation.py` `VGGT_DREAMER_SPECS[...]`
      assertions to read from the encoder classes.
- [ ] 5.3 Full test suite green; golden fixtures (1.1, 1.4) byte-identical to the
      `main` baseline.
- [ ] 5.4 Smoke one VGGT run and one CNN run end-to-end via
      `scripts/slurm/configs/*.yaml` smoke mode.
- [ ] 5.5 Load a **pre-change** checkpoint with `evaluate` and confirm
      `from_snapshot` resolves. The golden fixture proves the snapshot is stable;
      this proves the real recovery path works.
- [ ] 5.6 Resolve the two open questions in `design.md` (where `storage` belongs;
      whether a compat `variant` property survives) before archiving.
