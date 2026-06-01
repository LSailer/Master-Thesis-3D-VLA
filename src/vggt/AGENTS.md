# AGENTS.md — `src/vggt/`

Module contract for the VGGT 3D visual encoder. Scopes the repo-root
[`AGENTS.md`](../../AGENTS.md) to this package. Read before editing under `src/vggt/`.

## Purpose

VGGT (Visual Geometry Grounded Transformer) turns a stream of `518×518` RGB frames
into per-pixel **metric 3D points** (`world_points`) and a **camera pose**, plus the
aggregator tokens used as features. There are two implementations:

- a **PyTorch reference** wrapping the external StreamVGGT model, and
- a **JAX/Flax port** (`jax/`) with an explicit streaming KV-cache and dynamic budget
  eviction — this is the **production** path that feeds R2Dreamer's VGGT encoders.

It is consumed by `src/r2dreamer/adapters/vggt_adapter.py` and
`src/r2dreamer/encoders/__init__.py`.

## Layout

```
src/vggt/
├── feature_extractor.py  PyTorch VGGTFeatureExtractor — loads StreamVGGT from HF
│                         (lch01/StreamVGGT), per-frame extract() under no_grad,
│                         fixed 518×518 in → {world_points (37,37,3), camera_pose (9,)}
├── variants.py ......... VARIANTS registry + load_variant() (VGGT/StreamVGGT/InfiniteVGGT)
├── benchmark.py ........ run_inference(), benchmark_variant(), build_comparison_table()
├── plots.py ............ plot_comparison() (latency/memory charts)
│
└── jax/                  JAX/Flax port — PRODUCTION
    ├── feature_extractor.py  JAXVGGTFeatureExtractor — drop-in for the PyTorch API
    │                         (reset()/extract()); JIT'd per-frame; padded KV-cache;
    │                         configurable dtype/budget/pooling; warms up in __init__
    ├── aggregator.py ... Aggregator — 24 alternating frame/global attention blocks;
    │                     streaming (S=1/frame) + no-cache paths; emits 1374×2048 tokens
    ├── backbone.py ..... DinoV2Backbone, PatchEmbed — frozen ViT-L/14-reg, fixed 518²
    ├── block.py ........ Block, LayerScale, Mlp (pre-norm transformer block)
    ├── attention.py .... Attention — QK-norm, 2D RoPE, padded 3-tuple cache,
    │                     _padded_evict() (cosine-similarity pruning)
    ├── rope.py ......... compute_1d_rope_tables(), apply_rope_2d() (stateless)
    ├── weight_transfer.py load_checkpoint(), load_pytorch_weights() — torch→Flax
    │                     transposition; V1_EXCLUDE_PREFIXES skips depth/track heads
    ├── profile_streaming.py / benchmark_streaming.py  cache/latency profiling
    └── heads/
        ├── camera_head.py  CameraHead — AdaLN-modulated trunk, 4 refinement iters → 9-d pose
        └── dpt_head.py ... DPTHead — DPT decoder over layers 4/11/17/23 → 518×518×4 (pts3d+conf)
```

## Entry points

```python
# Production (JAX) — what R2Dreamer uses
from src.vggt.jax import JAXVGGTFeatureExtractor
ex = JAXVGGTFeatureExtractor(device="cuda", total_budget=200_000,
                             budgets_static=(8333,)*24, dtype=jnp.bfloat16,
                             compute_heads=True, wp_pool_size=37)
ex.reset()                                  # once per episode
out = ex.extract(rgb_518_uint8)             # {"world_points","camera_pose","aggregator_features"}
out = ex.extract(rgb_518_uint8, return_dense=True)   # + "dense_world_points" (518,518,3)

# Weight loading (HF StreamVGGT → Flax layout)
from src.vggt.jax import load_checkpoint, load_pytorch_weights
tree, scope = load_pytorch_weights(load_checkpoint(), include_v1_only=True)

# PyTorch reference / benchmarking
from src.vggt.feature_extractor import VGGTFeatureExtractor
```

## How R2Dreamer consumes it

- `VGGTObsAdapter` (`src/r2dreamer/adapters/vggt_adapter.py`) calls `extract()` per frame
  and pools outputs with `flatten_world_points_camera_pose()` (→ 4116-d WP/CP) or
  `pool_aggregator_tokens()` (cam + mean-patch + max-patch).
- Encoder variants (`src/r2dreamer/encoders/__init__.py`): `vggt` (WP/CP MLP),
  `vggt_aggregator_mlp` (pre-head tokens, `compute_heads=False`), `vggt_wp_dense_cnn`
  (dense 518² point map → conv), `vggt_wp_cp_64` (64×64 pool), `hybrid` (CNN + gated MLP).

## Architecture & cache

- **Tokens** per frame: `1 (camera) + 4 (register) + 37² (patches) = 1374`; channel dim
  `2048 = 1024 frame-stream ⊕ 1024 global-stream`. The global stream (`[..., 1024:]`) is
  the aggregator feature exposed to encoders.
- **Padded 3-tuple cache** `(k_pad, v_pad, valid_len)` keeps shapes fixed for JIT.
  Per-block budgets are computed **outside** JIT (Python ints, static args); eviction keeps
  anchor tokens + top-k by cosine-similarity-to-mean.
- **JIT compiles** are bounded: aggregator compiles twice (first frame vs subsequent),
  camera/point heads once each; `__init__` warms both cache states.

## Conventions

- **Fixed 518×518 input.** DINOv2 positional embeddings are hardcoded to the 37×37 grid;
  there is no bicubic-AA interpolation for other resolutions (a known follow-up).
- **Mixed precision.** Aggregator/camera head run in `bfloat16` (GPU cap ≥ 8); DPT uses
  fp32 intermediate convs and emits the final point map in fp32 for geometric stability.
- **torch→Flax weight rules** (`weight_transfer.py`): Conv2d `(O,I,H,W)→(H,W,I,O)`;
  ConvTranspose2d transposes **and spatially flips**; Linear `(O,I)→(I,O)`; LayerNorm unchanged.
  `depth_head.`/`track_head.` are excluded by `V1_EXCLUDE_PREFIXES`.
- **Streaming API contract:** `reset()` at episode start; `extract()` takes `(3,518,518)` uint8.

## Dependencies

- JAX path: `jax`, `flax.linen`, `numpy`, `huggingface_hub` (checkpoint `lch01/StreamVGGT`).
- PyTorch path: `torch`, `external/InfiniteVGGT/src/streamvggt` (StreamVGGT reference).
- Consumers: `src.r2dreamer.adapters.vggt_adapter`, `src.r2dreamer.encoders`.

## Running & testing

VGGT touches the GPU — always `srun` (see root `AGENTS.md`). The real extractor is
exercised by GPU-marked tests:

```bash
srun --partition=dev_gpu_h100 --gres=gpu:1 --time=00:30:00 \
  uv run pytest tests/vggt/ -q
```

## Gotchas / read-this-first

- **JAX vs PyTorch must stay bit-comparable.** The ConvTranspose spatial-flip and fp32 DPT
  convs exist for parity; changing them silently breaks 3D accuracy. Validate with
  `jax/benchmark_streaming.py` before/after.
- **Camera-head frame cap.** The camera cache is fixed-window (`max_camera_frames × iters`);
  overrunning it raises `RuntimeError`, not a soft truncation.
- **Aggregator tokens are 2048-d (frame⊕global)** — encoders take the global half. Don't feed
  the full 2048-d vector where 1024-d is expected.
- **Budget/eviction are static JIT args.** Passing per-call Python-int budgets avoids
  recompilation; threading a traced value triggers a recompile storm.
- **Compilation cache** defaults under `/tmp`; override `JAX_COMPILATION_CACHE_DIR` if `/tmp`
  is not writable on the node.
