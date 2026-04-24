# VGGT JAX port — Step 8 initial benchmark

`modules/vggt/jax/benchmark_streaming.py`, 10-frame episode on H100, synthetic uint8 frames.

| Backend | Config | Mean ms/frame | Median ms/frame | Peak mem |
|---|---|---|---|---|
| PyTorch | bf16 autocast (production) | 98.5 | 94.9 | 8.0 GB |
| JAX     | eager fp32 (no jit)        | 5395.9 | 5959.6 | n/a |

JAX is ~55× slower than PyTorch at this configuration. The gap is expected given:

- JAX runs eager — the Python graph is rebuilt per frame; no kernel fusion.
- JAX runs fp32 — PyTorch uses bf16 autocast, which on Ampere+ GPUs is 2-3× faster.
- PyTorch's aggregator uses `F.scaled_dot_product_attention` (fused SDPA); our JAX attention is a manual einsum.

To meet the plan's "not slower than PyTorch compile + bf16" latency floor, follow-up work needs:

1. `jax.jit` wrap of the per-frame forward in `JAXVGGTFeatureExtractor.extract`.
2. Bf16 autocast for aggregator matmuls with fp32-preserved softmax and fp32 heads.
3. Optionally a Pallas or flash-attention kernel for the global blocks with large KV caches.

The benchmark CSV lives at `output/comparison/` (gitignored). Re-run with:

```bash
uv run python -m modules.vggt.jax.benchmark_streaming --seq-lens 10 50 100
```
