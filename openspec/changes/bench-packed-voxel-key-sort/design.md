## Context

`_add_frame_to_state` (jitted, `src/buffer/house_context_pose_buffer.py`)
dedupes each incoming ~268k-row VGGT frame by sorting rows on their voxel
key. The sort is `jnp.lexsort((k2, k1, k0, ~valid))` — a single variadic
`lax.sort` with 4 key operands plus the implicit iota value operand. XLA's
GPU sort rewriter dispatches to CUB `DeviceRadixSort` only for simple 1–2
operand sorts; more operands fall back to the O(n·log²n) bitonic comparator
sort. Estimated traffic: ~3 GB/frame (comparator) vs ~200 MB (radix pairs).
The estimate has not been measured; the project rule is that optimizations
land only on benchmark evidence (`scripts/r2dreamer/bench_graph_vs_buffer.py`
times exactly this kernel on realistic data).

Voxel keys are `floor(world_m / voxel_size_m)` as int32 — full int32 range in
type, but physically bounded by house extents (a few tens of metres).

## Goals / Non-Goals

**Goals:**
- Measure, on the target GPU, whether a packed uint64 single-key sort beats
  the current 4-key lexsort in the buffer-add kernel.
- Keep dedupe semantics exactly equivalent within a documented key range.
- Land the faster variant only on a clear measured win; otherwise keep the
  lexsort and record the numbers.

**Non-Goals:**
- No sort-free redesign of the probe loop (scatter-min dedupe) — separate
  question, only relevant if the sort remains the hotspot after this change.
- No change to the hash table, probe loop, snapshot path, or public buffer
  API.
- No CPU optimization — the question is specifically the GPU sort path.

## Decisions

- **Pack layout**: 21 bits per axis, biased: `bK = clamp-or-invalidate(kK) +
  2^20`, `packed = (b0 << 42) | (b1 << 21) | b2`, with `uint64::max` reserved
  as the invalid sentinel (non-finite points, low-confidence points, and keys
  outside ±(2^20 − 1)). Why: a general 96→64-bit pack is impossible; 21 bits
  per axis covers ±10.5 km world extent at 1 cm voxels — orders of magnitude
  beyond any house scene — and leaves the sentinel unambiguous. Alternative
  considered: 32/16/16 asymmetric split — rejected, no axis is privileged.
- **Out-of-range keys are invalidated, not clamped**: clamping would merge
  distant garbage points into one boundary voxel; routing them to the
  sentinel treats them like non-finite points, which they effectively are.
- **Sentinel replaces the `~valid` sort key**: invalid rows get
  `uint64::max`, so validity ordering falls out of the single key. After the
  sort, `sorted_valid = packed_sorted != sentinel` and `same_as_previous`
  is a scalar compare — the 3-column `jnp.all` disappears. `key_xyz` is still
  gathered for the hash-table probe (exact-key compare there is unchanged).
- **A/B selection via a benchmark flag, not a code fork**: the bench script
  gains `--sort-variant {lexsort,packed}`; the kernel keeps one
  implementation per variant behind a small helper so the jit graph is
  identical apart from the sort. Both variants stay in the branch until the
  decision, then the loser is deleted.
- **Decision threshold**: "clear win" = the packed variant reduces steady
  state buffer-add time by a margin that survives noise (rule of thumb:
  >15% median improvement over ≥50 timed iterations). Below that, keep the
  simpler-typed lexsort.

## Risks / Trade-offs

- [XLA may already fuse/dispatch differently on the cluster's version] →
  the benchmark is the arbiter; also capture an HLO dump
  (`XLA_FLAGS=--xla_dump_to`) in one run to confirm which sort path is taken.
- [Packed variant wins on GPU but regresses CPU tests/dev machines] →
  equivalence tests are dtype/platform-independent; CPU perf is a non-goal,
  but note it in the result summary.
- [Silent precondition violation later (huge coordinates)] → out-of-range
  keys are invalidated (dropped), same as NaN points today, and the bound is
  documented at the pack site; optionally count them into
  `failed_insert_count`-style diagnostics if review wants visibility.
- [uint64 needs jax x64 or careful dtype handling] → JAX supports uint64
  arrays without global x64 only in limited ways; if `jax_enable_x64` is off,
  uint64 ops may silently downcast. Verify dtype end-to-end in the
  equivalence test; fall back to two uint32 keys (still 2 operands, still
  radix-eligible) if uint64 proves awkward.

## Open Questions

- Does the cluster's XLA version radix-sort a 2-operand (key, iota) uint64
  sort? (Answered by the HLO dump in the first benchmark run.)
- Is `failed_insert_count`-style accounting for out-of-range keys wanted, or
  is silent drop (like NaN) fine?
