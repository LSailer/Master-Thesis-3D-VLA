# Reference index — VGGT cache persistence

> Pointer to the original source files touched by the work described in
> `handoff.md`. Line numbers are as of commit on `develop` (2026-07-02); they
> drift, so search by symbol name. All paths are repo-relative.

## Code we added or changed (this work)

### `src/vggt/jax/feature_extractor.py` — the JAX streaming extractor
| Symbol | Line | Role |
|---|---|---|
| `_DEFAULT_TOTAL_BUDGET = 1_200_000` | 49 | extractor default; now matches the training encoder + InfiniteVGGT |
| `class ResetMode(Enum)` | 97 | `FULL` (default, per-episode wipe) / `PERSIST_SCENE` (per-scene save/restore) |
| `class AggregatorCacheSnapshot` | 114 | frozen holder for the 4 streaming fields |
| `__init__(…, reset_mode=ResetMode.FULL)` | 254, 260 | wires mode + `_scene_cache_store` / `_current_scene_id` |
| `_configure_camera_cache` / `_CAM_MAX = max_camera_frames * _cam_num_iters` | 344, 349 | fixed padded camera buffer shape |
| `reset()` | 533 | the pure wipe primitive (unchanged) |
| `save_cache()` | 540 | snapshot the 4 fields (JAX array refs, no device copy) |
| `load_cache(snap)` | 560 | restore the 4 fields verbatim |
| `reset_for_scene(scene_id)` | 577 | mode-aware entry: `FULL`→`reset()`; `PERSIST_SCENE`→save outgoing / restore-or-init incoming |
| `_check_camera_cache_capacity()` | 672 | **was raise → now calls evictor** (sliding window) |
| `_evict_oldest_camera_frame()` | 693 | shift each trunk block left by `_cam_num_iters` rows, pad tail, `valid_len = _CAM_MAX − n` |

Also in this file (unchanged, but the eviction relies on them):
- `_run_heads` calls `_check_camera_cache_capacity()` immediately before the
  jit-compiled `camera_head_apply` — so the order is evict→apply→`valid_len`
  back to `_CAM_MAX`.
- the camera-head trunk blocks run with `cache_budget=None` (camera_head.py),
  which is **why** the extractor-level evictor is needed — unlike the
  aggregator, the camera head has no internal eviction.

### `src/r2dreamer/encoders/base.py` — the training encoder config
| Symbol | Line | Change |
|---|---|---|
| `VGGT_TOTAL_BUDGET = 1_200_000` | 152 | was `200_000`; raised to the InfiniteVGGT default for a fair token-space comparison |
| `VGGT_STATIC_BUDGETS = tuple([50_000] * 24)` | 153 | was `(8333,)*24`; uniform 50k/block (1.2M ÷ 24) |
| `total_budget=` / `budgets_static=` passed to the extractor | 205, 206 | — |

## Reference behaviour we matched (read-only, external)

- `external/InfiniteVGGT/src/streamvggt/models/streamvggt.py:19` —
  upstream default `total_budget=1200000` (the number we now match).
- `external/InfiniteVGGT/src/streamvggt/heads/camera_head.py` — the model's
  camera head grows `past_key_values_camera` **unbounded** (no internal cap;
  each iteration appends block KV). The cap is the wrapper's job, not the
  model's.
- `src/vggt/reference/feature_extractor.py:209-217` — the PyTorch reference
  wrapper implements the sliding window we mirrored: when
  `max_camera_frames` is set it slices `k[:, :, -max_camera_tokens:, :]`
  (keep most-recent N). Default `max_camera_frames=None` → unbounded. This
  is the semantics `_evict_oldest_camera_frame` reproduces in the JAX port.

## The eviction machinery already in the tree (aggregator uses it; camera head does not)

- `src/vggt/jax/attention.py` — padded-cache path:
  - `:267-274` append new K/V at offset `valid_len` via
    `dynamic_update_slice_in_dim`, `new_valid_len = valid_len + N`.
  - `:280-290` `jax.lax.cond(new_valid_len > cache_budget, evict, passthrough)`
    — **diversity** eviction (keep anchors + lowest-cosine-sim-to-mean).
  - The aggregator passes `cache_budget` per block (so it evicts); the
    camera head passes `None` (so it does not, and the extractor evictor is
    the bound). We deliberately did **not** reuse the diversity evictor for the
    camera head — the reference uses a *recent-window*, not diversity.

## Where `reset()` is wired (unchanged — the §7 step-3 wiring site)

- `src/vggt/jax/feature_extractor.py` `_image_from_extract_input` —
  `is_first=True` → `self.reset()` (the live self-reset path).
- `src/r2dreamer/adapters/vggt_adapter.py` and
  `src/r2dreamer/adapters/hybrid_adapter.py` — `on_episode_reset=extractor.reset`
  (some paths pass `None`, the never-reset regime). Step 3 replaces these
  with `reset_for_scene(scene_id)` behind a flag.

## Companion docs

- `docs/notes/cache-persistence/handoff.md` — the design + status (this
  folder's canonical handoff).
- `src/prototyp/cache_persistence/EXPERIMENTS.md` — the E0–E3 experiment plan
  (kept with the future drivers under `src/prototyp/`, per its `AGENTS.md`).
- `docs/notes/house-context-episode-reanchoring.md` — the misalignment fix
  (Option C, GT-pose Umeyama) — a *different* lever (the point buffer), read
  alongside this handoff.
- `docs/notes/house-context-misalignment-problem.md` — the plain-language bug.
- `docs/notes/house-context-full-buffer-options.md` — buffer design options.
- `docs/notes/2026-07-02-graph-house-context-status.md` — work-stream status
  map.
- `docs/notes/graph-spectral-house-context-experiment.md` — graph-spectral
  house-context experiments.