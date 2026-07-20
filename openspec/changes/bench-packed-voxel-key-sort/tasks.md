## 1. Packed-key variant

- [ ] 1.1 Add a `_pack_voxel_keys` helper in
      `src/buffer/house_context_pose_buffer.py`: bias each int32 key
      component by +2^20, route components outside ±(2^20 − 1) and invalid
      rows to the all-ones uint64 sentinel, pack as
      `(b0 << 42) | (b1 << 21) | b2`; document the key-range precondition in
      its docstring
- [ ] 1.2 Verify uint64 dtype survives end-to-end without `jax_enable_x64`
      (check the packed array's dtype under jit); if it downcasts, switch to
      the two-uint32-key fallback from design.md
- [ ] 1.3 Add the packed sort path to `_unique_frame_voxels` behind a static
      variant flag: single-key sort on the packed key,
      `sorted_valid = packed_sorted != sentinel`, scalar `same_as_previous`
      compare; keep `key_xyz` gather for the probe loop unchanged

## 2. Equivalence tests

- [ ] 2.1 Extend `tests/buffer/test_house_context_pose_buffer.py` with a
      randomized equivalence test: both variants produce the identical
      active (key → representative XYZ/RGB) map on frames with duplicates,
      NaN rows, and sub-threshold confidence rows
- [ ] 2.2 Add an out-of-range key test: points beyond ±(2^20 − 1) voxels per
      axis are invalidated by the packed variant (never active), matching
      non-finite handling

## 3. Benchmark

- [ ] 3.1 Add `--sort-variant {lexsort,packed}` to
      `scripts/r2dreamer/bench_graph_vs_buffer.py`, threading the static
      variant flag into the jitted buffer-add kernel
- [ ] 3.2 Run both variants on the target GPU (SLURM) with identical seed and
      frame size, ≥50 timed iterations each; record median and spread
- [ ] 3.3 Capture one run with `XLA_FLAGS=--xla_dump_to` and confirm from the
      HLO which sort path (CUB radix vs comparator) each variant takes

## 4. Decision and cleanup

- [ ] 4.1 Apply the decision gate: packed wins only on >15% median
      steady-state improvement; record the numbers and the HLO finding in
      this change
- [ ] 4.2 Remove the losing variant and the variant flag from kernel and
      benchmark; if packed won, keep the precondition documentation at the
      pack site
- [ ] 4.3 Run the full buffer test suite and the benchmark once more on the
      surviving implementation
