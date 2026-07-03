# Visible house-context snapshot (accumulate all, feed what the camera sees)

**Status:** review + design only (2026-07-03). A first implementation was
built, tested, and then **reverted on request** — no code changes are in the
tree. This note preserves the review findings and the validated design.

## Idea

The house map keeps accumulating **every** deduplicated VGGT point
(unchanged), while the `(max_points, 6)` snapshot handed to the encoder would
contain **only the stored points inside the current frame's camera frustum**,
instead of an insertion-order stride over the whole map.

## Deep review findings (3-subagent review, 2026-07-03)

### Layer 1 — VGGT KV cache (commits d3705ae, 9a40f5a)
- `ResetMode`/`AggregatorCacheSnapshot`/`reset_for_scene` snapshot all four
  streaming fields **including** `_past_kvs_camera` and `_frame_idx`
  (`src/vggt/jax/feature_extractor.py:540-604`). Adapter wiring (handoff §7
  step 3) is still pending.
- The camera-head sliding-window eviction (§4.2b) exists **only as an
  uncommitted diff** in `feature_extractor.py`, yet
  `docs/notes/cache-persistence/handoff.md` marks it ✅ DONE. Commit it.
- Stale docstrings: `ResetMode` (:104) and `reset_for_scene` (:588) still
  claim the camera cache is unbounded / raises — no longer true.
- Dead code: `attention.py:295-301` assigns `did_evict` with a wrong formula
  that line 302 immediately overwrites; harmless, delete.
- Zero headroom: training's static budget 50 000 == `_MAX_BUDGET` works
  exactly (padded max 51 374 = 50 000 + P), but any frame contributing > P
  tokens would silently clamp-overflow — the static path has no runtime guard.
- Camera eviction costs a host sync + eager concat per trunk block per frame
  once full (R5); a circular buffer would avoid it.

### Layer 2 — house-context buffer pipeline
- `HouseContextPoseBuffer` (`src/buffer/house_context_pose_buffer.py`) is a
  fixed-shape jitted voxel hash (training: capacity 2^23, table 2^24 via
  adapter overrides). No reset — persists across episodes by design.
- Before this change the encoder snapshot was the whole map (verbatim prefix
  or discovery-time stride); **no visibility concept existed on the training
  path**. The only prior "visibility" was the prototype's NN-based
  `visible_steps` diagnostic (`src/prototyp/nearestNeighborComparision.py`).
- `augment_replay_batch` broadcasts the *latest* rollout snapshot to all
  replay timesteps (single-scene approximation, documented in the adapter).
- Known open issues that interact with visibility: cross-episode
  misalignment (ghost copies; 3.38 M pts in 1200 steps, job 5706115) and
  bfloat16 `store_xyz` position aliasing beyond ~2.56 m.

### Layer 3 — visibility primitives
- Camera head emits a 9-D `absT_quaR_FoV` encoding: `[:3]` translation,
  `[3:7]` quaternion **scalar-last (XYZW)**, `[7:9]` `(fov_h, fov_w)`.
  Extrinsic `[R|t]` is **world-to-camera, OpenCV** (x-right, y-down,
  z-forward). Intrinsics derive from FoV: `fx = (W/2)/tan(fov_w/2)`,
  `fy = (H/2)/tan(fov_h/2)`, principal point at the image centre
  (reference: `external/VGGT/vggt/utils/pose_enc.py:62-124`).
- `world_points` are already world-frame XYZ (no unprojection step), so
  visibility is pure per-point arithmetic over `store_xyz`.
- The decode utilities existed only in torch under `external/`; they are now
  ported to JAX (below).

## Validated design (built once, tests passed, then reverted)

| Piece | Where it would live |
|---|---|
| Pose decode (JAX port) | `src/vggt/jax/pose_enc.py` — `quat_to_rotation_matrix`, `decode_pose_encoding` → `CameraParams` |
| Visibility snapshot | `src/buffer/house_context_pose_buffer.py` — `_visible_context_snapshot` (jitted) + `visible_context_array()` |
| Shared stride/pad tail | `_strided_store_snapshot` — factored out of `_house_context_snapshot`, reused by both paths |
| Adapter wiring | `src/r2dreamer/adapters/hybrid_adapter.py` — `visible_only: bool = True` ctor flag; `transform` passes the raw float32 pose |
| Tests | `tests/vggt/test_pose_enc.py`, `tests/buffer/test_visible_context_snapshot.py` |

Per-point test (all inside one jit, O(capacity), no tree):

1. Decode pose → `R, t, fx, fy, cx, cy` (float32; bf16's ~2 px resolution at
   518 px is too coarse for the boundary test).
2. `p_cam = p_world @ R.T + t`; in-front: `z > 1e-6`.
3. `u = fx·x/z + cx`, `v = fy·y/z + cy`; in-image: `0 ≤ u < W`, `0 ≤ v < H`
   with `(H, W) = (518, 518)` (the VGGT input the FoV refers to).
4. Compact visible rows in insertion order via cumsum-rank scatter (a
   `(capacity,)` int32 lookup, ~33 MB at 2^23 — no `(capacity, 6)` scratch),
   then reuse the stride/pad tail → fixed `((max_points, 6), count)`.
   Zero visible points → `count = 0` → the encoder's existing masked pooling
   emits its "no-map" token.

The encoder contract (`HOUSE_CONTEXT_KEY` + `HOUSE_CONTEXT_SIZE_KEY`,
masked pooling) is unchanged; no encoder edits were needed.

## Caveats / follow-ups

- **No occlusion (v1):** a point behind a wall but inside the frustum counts
  as visible. If needed later: scatter-min z over projected `(u, v)` bins —
  still jittable, mirrors the buffer's `winner_by_slot` min-scatter.
- **Misalignment prerequisite:** the frustum test assumes map and pose share
  one world frame. Under per-episode `extractor.reset()`, points from earlier
  episodes live in other VGGT frames, so they are culled only approximately
  until re-anchoring (`house-context-episode-reanchoring.md`) or
  `ResetMode.PERSIST_SCENE` wiring lands. Visibility filtering *reduces* the
  ghost-copy blast radius at the encoder but does not fix the map.
- **Replay approximation sharpens:** replay batches now receive the context
  visible from the *latest rollout pose*, while their stored `camera_pose` is
  from another time. Egocentric context per replayed pose would require
  storing poses per step and snapshotting per batch element (cost: one
  projection pass per replay sample).
- bf16 `store_xyz` aliasing (> 2.56 m) adds noise at frustum edges — same
  pre-existing issue as the map itself.
- Separate pre-existing issue (still open after the revert):
  `tests/r2dreamer/launch/test_encoders.py::test_vggt_encoder_uses_static_jax_budgets`
  still asserts the old 200k budget from before 9a40f5a and fails; it needs
  updating to 1.2M / 50k×24.
