## Why

The per-frame voxel dedupe in `src/buffer/house_context_pose_buffer.py`
(`_unique_frame_voxels`) sorts every frame with a 4-key `jnp.lexsort`
(validity flag + 3 int32 voxel-key columns). A multi-operand `lax.sort`
likely misses XLA-GPU's CUB radix-sort fast path and falls back to the
O(n·log²n) bitonic comparator sort — estimated ~3 GB of memory traffic per
268k-row frame versus ~200 MB for a single-key radix `SortPairs`. That would
make the sort the dominant cost of the buffer-add kernel, which runs at every
environment step. The claim is theoretical; per project policy, optimizations
must be proven in-benchmark before landing.

## What Changes

- Add a packed-key variant of `_unique_frame_voxels`: pack the three int32
  voxel-key components into one uint64 sort key (bias each component by
  +2^20; clamp/route keys outside ±(2^20 − 1) to an all-ones invalid
  sentinel, like non-finite points; pack as `(b0<<42) | (b1<<21) | b2` with
  `uint64::max` reserved for invalid rows), then sort on that single key.
- Benchmark baseline (4-key lexsort) vs packed variant with the existing
  `scripts/r2dreamer/bench_graph_vs_buffer.py` buffer-add timing on the
  target GPU, at the real frame size.
- Decision gate: land the packed variant only on a clear measured win;
  otherwise record the numbers and keep the lexsort. Nothing lands blindly.
- If the packed variant wins, take the simplifications that fall out:
  `sorted_valid` becomes `packed != sentinel`, and `same_as_previous` becomes
  a scalar compare instead of `jnp.all` over 3 columns.
- Document the key-range precondition: a general 96→64-bit pack is lossy, so
  the ±(2^20 − 1) per-axis bound (±10.5 km world extent at 1 cm voxels) must
  be stated where the pack happens.

## Capabilities

### New Capabilities
- `voxel-dedupe-sort-benchmark`: measured comparison of the buffer-add
  dedupe sort (4-key lexsort baseline vs packed uint64 single-key sort),
  including equivalence checks and the benchmark-gated landing decision.

### Modified Capabilities

<!-- none — main specs are empty; buffer behavior requirements are unchanged
     (the pack must be exactly equivalent within the documented key range) -->

## Impact

- `src/buffer/house_context_pose_buffer.py`: `_unique_frame_voxels` (and, on
  a win, the small follow-on simplifications inside it). Kernel signature and
  `_UniqueFrameVoxels` output stay unchanged.
- `scripts/r2dreamer/bench_graph_vs_buffer.py`: benchmark entry point; may
  gain a flag to select the sort variant for A/B timing.
- `tests/buffer/test_house_context_pose_buffer.py`: equivalence coverage for
  the packed variant (same representative per voxel as the lexsort path).
- Requires a GPU run (SLURM cluster) — CPU timings do not answer the CUB
  radix-path question.
