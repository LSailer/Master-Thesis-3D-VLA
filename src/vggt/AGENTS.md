# AGENTS.md — `src/vggt/`

VGGT package rules. Inherits repo-root `AGENTS.md`.

## Purpose

VGGT converts a stream of fixed `518×518` RGB frames into per-pixel metric
`world_points`, a `camera_pose`, and aggregator-token features. The JAX/Flax port
in `jax/` is the production path; `reference/` wraps the PyTorch StreamVGGT model.

## Public contract

- Call `reset()` at every episode boundary.
- `extract()` takes `(3, 518, 518)` `uint8` CHW RGB.
- With heads enabled, output includes dense `world_points` and `camera_pose`.
- With `compute_heads=False`, use aggregator tokens only.
- Aggregator tokens are `2048 = 1024 frame ⊕ 1024 global`; encoders usually consume
  the global half.

## Architecture constraints

- Input size is fixed: DINOv2 positional embeddings assume a `37×37` patch grid.
- Streaming cache is padded `(k_pad, v_pad, valid_len)` for stable JIT shapes.
- Budget/eviction values are Python-int static JIT args; do not pass traced budgets.
- Aggregator/camera run mixed precision; DPT uses fp32 conv intermediates and emits
  fp32 point maps.
- Weight transfer rules in `jax/weight_transfer.py` are correctness-critical,
  especially ConvTranspose spatial flip and excluded V1 prefixes.

## R2Dreamer integration

- Observation adapters in `src/adapters/` hold the extractor and call `extract()`:
  `global_tokens.py` (global/full/pooled aggregator tokens), `pointmap_pose.py` and
  `pointmap_dense.py` (WP/CP and dense point map), `house_voxels.py` and
  `house_cloud_episodes.py` (accumulated world points).
- An adapter declares its model composition by returning `AdapterField(key, encoder, ...)`
  values, where `encoder` is a member of the `Encoder` enum in `src/adapters/contract.py`.
  That routing - not a config string - selects the branch modules in
  `src/r2dreamer/encoders/`.

## Gotchas

- JAX and PyTorch paths must stay bit-comparable; validate parity-sensitive changes with
  `jax/benchmark_streaming.py`.
- Camera cache is fixed-window (`max_camera_frames × iters`); overflow raises `RuntimeError`.
- Do not feed full 2048-d aggregator tokens where 1024-d global tokens are expected.
- JAX compilation cache defaults to `/tmp`; set `JAX_COMPILATION_CACHE_DIR` if needed.

## Running/testing

VGGT requires GPU execution under `srun`:

```bash
srun --partition=dev_gpu_h100 --gres=gpu:1 --time=00:30:00 \
  uv run pytest tests/vggt/ -q
```
