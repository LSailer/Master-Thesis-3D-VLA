# VGGT JAX Streaming — Padded 3-tuple KV-cache + jit

**Date**: 2026-04-24
**Closes**: [#81](https://github.com/LSailer/Master-Thesis-3D-VLA/issues/81), builds on [#80](https://github.com/LSailer/Master-Thesis-3D-VLA/pull/80) (aggregator jit)
**Touch points**: `src/vggt/jax/feature_extractor.py`, `src/vggt/jax/attention.py`, `src/vggt/jax/heads/camera_head.py`, `src/vggt/jax/aggregator.py`

## Motivation

The JAX port of StreamVGGT was ~24× slower than PyTorch despite using identical math and weights. Root cause: streaming attention with a growing KV-cache is the dominant phase, and a legacy 2-tuple cache (`(k, v)`) grows its shape every frame, forcing JAX to recompile the graph per frame. With 24 aggregator blocks × 4 camera-head iterations × 4 trunk blocks × N frames, the compile overhead dwarfs compute.

Commit f30273b jitted the aggregator this way; commit 90e123d extended the same pattern to `camera_head`. After both, end-to-end JAX extract is within 1.9× of PyTorch (137 ms vs 72 ms per frame on H100 bf16).

## The Pattern — padded 3-tuple + single jit'd graph

Instead of a growing `(k, v)` cache, we allocate a fixed-shape buffer up front and track how much of it is valid:

```python
past_kv = (k_pad, v_pad, valid_len)  # shapes: (B,H,MAX,Dh), (B,H,MAX,Dh), int32 scalar
```

The attention module routes on tuple length:

- `len == 2` → legacy path (`jnp.concatenate` + manual einsum softmax). Kept for fp32 parity tests.
- `len == 3` → padded path: `jax.lax.dynamic_update_slice_in_dim` writes new K/V at offset `valid_len`; attention uses `jax.nn.dot_product_attention` with `key_value_seq_lengths=valid_len` (cuDNN flash kernel for bf16/fp16, XLA with bool mask for fp32).

Because `k_pad`, `v_pad`, `valid_len` have fixed shapes, **one compiled graph covers all frames**. Only `valid_len` (an int32 scalar) changes.

### Aggregator specifics (PR #80)

- `_MAX = (total_budget // depth) + P` where `P = 5 + (518/14)^2 = 1374` patch tokens per frame.
- Eviction fires inside the jit'd graph via `jax.lax.cond`: `_padded_evict` computes cosine-similarity scores over valid slots, keeps top-k, zeroes the rest.
- Two compiled graphs (`is_first_frame: bool` static arg): first-frame has no KV to attend to; subsequent frames do. Without this split, frame-0 attention masks would differ in shape from frame-N.

### Camera head specifics (#81)

- `_CAM_MAX = max_camera_frames × num_iterations` (default 1024 × 4 = 4096 slots per trunk block).
- **No eviction** — camera head keeps all frames' K/V within an episode. `_CAM_MAX` just has to be large enough for the longest episode.
- **One compile, not two.** Unlike the aggregator, camera_head's graph doesn't depend on `is_first_frame` because `valid_len=0` already handles the first frame correctly in the padded path.
- Each frame appends `num_iterations = 4` tokens per trunk block (not 1), because each iteration of the iterative pose refiner appends its own K/V.

## Overflow guard

`jax.lax.dynamic_update_slice_in_dim` **silently clamps** start indices to fit within the destination buffer. `jax.nn.dot_product_attention` with `key_value_seq_lengths > MAX` has undefined cuDNN behavior. Result: if `valid_len` ever exceeds `_CAM_MAX`, new writes overwrite the last slot and reads go out of bounds — no exception, just wrong numbers.

Solution (added in #81): `feature_extractor.py::extract()` checks `self._frame_idx >= max_frames` before the camera apply and raises `RuntimeError` with a helpful message. Locked by `test_jax_integration.py::test_camera_cache_overflow_raises`.

## Verifying parity under the change

The existing test matrix covers this pattern thoroughly:

- `TestLevel3PaddedCacheParity` — locks aggregator padded vs legacy bit-parity (atol 1e-5).
- `TestLevel3CameraHeadPaddedParity` (added in #81) — same for camera_head.
- `TestLevel3CameraHeadCache` / `TestLevel2CameraHead` — camera head PT↔JAX parity (legacy path).
- `test_jax_integration::test_rollout_parity` — end-to-end PT vs JAX on 5-frame rollouts.

Recommended targeted regression check after any aggregator/camera-head change:
```
pytest tests/vggt/test_jax_integration.py \
       tests/vggt/test_jax_parity.py::TestLevel2CameraHead \
       tests/vggt/test_jax_parity.py::TestLevel3CameraHeadCache \
       tests/vggt/test_jax_parity.py::TestLevel3CameraHeadPaddedParity \
       tests/vggt/test_jax_parity.py::TestLevel3PaddedCacheParity \
       -v --tb=short
```
~7 min on H100.

## Deferred / known gaps

- **Phase 2: shared helper.** The padded-cache bookkeeping is duplicated between aggregator and camera_head (`_new_padded_cache_entry` / `_new_padded_camera_entry`, `_to_padded`, `_MAX` / `_CAM_MAX`). Extracting a `src/vggt/jax/padded_cache.py` helper is a clean refactor but was deferred from #81 to keep the perf change small.
- **point_head jit** — already <1% of extract (2.7 ms). Low ROI.
- **bf16 parity tolerance tightening** — current L1 dump shows maxabs 1.1e-2 (rel_err 5e-3) PT↔JAX, which is the bf16 streaming-attention precision band. If a downstream metric turns out to be sensitive to <1e-2 drift, revisit.

## Numbers (bf16, H100, n=10 frames)

| phase | pre-#81 | post-#81 |
|---|---|---|
| input_prep | 1.3 ms | 1.4 ms |
| aggregator (jitted) | 54 ms | 55 ms |
| **camera_head** | **1187 ms (eager)** | **4.3 ms (jitted)** |
| point_head | 2.7 ms | 2.6 ms |
| pool + host transfer | 0.9 ms | 0.9 ms |
| **TOTAL** | **1247 ms** | **64 ms** |

End-to-end `bench_fast` (JAX median): 1763 ms → 137 ms. Speedup vs PT: 0.041× → **0.524×**.

## Related

- [#74 L4 Pipeline Profiling](l4-profiling.md) — Original motivation to port VGGT to JAX.
- [#78 PRD: Autoresearch harness](https://github.com/LSailer/Master-Thesis-3D-VLA/issues/78) — Surrounding autoresearch infrastructure.
- [#82 Alternate speedup strategies](https://github.com/LSailer/Master-Thesis-3D-VLA/issues/82) — Non-JAX-port speedups (resolution, distillation, etc.), complementary.
