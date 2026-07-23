# Known risks / open points

- **Optimizer-state checkpoint layout changes** with three LaProp states.
  Old checkpoints must keep loading (shim in `checkpointing.py`); verify with
  `tests/r2dreamer/test_checkpointing.py` plus a new three-state case.
- **Golden-run sensitivity:** any reordering of `jax.random.split` calls or
  of env/buffer operations breaks CSV equality. If a step cannot preserve
  order, stop and split that change out — do not accept a non-identical run.
- **Gate-fusion parity:** `WMHybridEncoder` has structured replay/live layout
  handling; the CompositeEncoder gate must reproduce it bit-exactly before
  the hybrid golden run can pass.
- **Lazy obs-spec inference needs a real first frame** (env + extractor
  loaded). Unit tests must feed the adapter a fake frame instead; never load
  VGGT weights in CPU tests.
- **Unported encoder types** stay on the old factory path during migration —
  keep `_resolve_encoder_cls` alive until the last type is ported.
