# Open problems + decision log — live house context PERSIST

## Decision log

| # | Decision | Rationale | Date |
|---|---|---|---|
| D1 | Scene-aware `on_episode_reset(scene_id)` at all 4 trainer boundaries; capture the reset obs prefill discarded | DreamerV3 emits `is_first` at every episode start; prefill discarding the reset frame meant the boundary signal never reached the VGGT cache (PROTOCOL §3.D1) | 2026-07-03 |
| D2 | Backward-compatible `lambda scene_id="scene": ...` (default arg) | ~11 no-arg callers in profiling/eval scripts; default-arg is the standard compat pattern, avoids high-churn signature sweep (PROTOCOL §3.D2) | 2026-07-03 |
| D3 | Verify with IoU/%new-voxels A/B, not growth curve | Growth can't separate ghost-copies from legitimate exploration; alignment is the defining property of persistent scene memory (RIM/SSM/GSMem) (PROTOCOL §3.D3) | 2026-07-03 |
| D4 | Prevent re-anchor (PERSIST) over correct-after-the-fact (Umeyama) | Cheaper + more robust; option C stays a fallback (PROTOCOL §3.D4) | 2026-07-03 |

## Open problems / risks

- **A2 (unverified):** does restoring the VGGT KV cache actually keep VGGT in
  the same world frame, or does the camera head re-estimate pose from scratch
  ignoring the cache? The IoU diagnostic settles this. If PERSIST does NOT
  improve alignment, the fix is option C (GT-pose + running Umeyama) and
  PERSIST is insufficient on its own.
- **A1 (diagnostic confound):** the %new-voxel signal is only clean when
  episodes overlap (same spawn). L1 single-episode curriculum should give the
  same spawn on `env.reset()`, but if Habitat cycles episodes, %new could be
  high under correct PERSIST. Mitigation: the A/B vs FULL controls for this —
  both modes see the same spawns, so a PERSIST≪FULL %new difference is
  attributable to frame consistency, not exploration.
- **bf16 `store_xyz` aliasing beyond ~2.56 m** — pre-existing; adds noise at
  frustum edges / far points. Not addressed here; revisit if the diagnostic
  shows residual misalignment at large radii.
- **Camera-head eviction cost (R5)** — once `max_camera_frames` fills under
  PERSIST, a per-frame concat per trunk block. Watch step time vs the 158 ms
  baseline in the smoke; lower `max_camera_frames` if it regresses.
- **Multi-house curriculum** — PERSIST keys by `scene_id`; an empty scene_id
  falls back to `"scene"`, which would mix houses. Fine for L1 (single house);
  needs real scene_ids for L2+.

## Dead ends (not pursued)

- **Make prefill process the reset frame through `transform`** — would trigger
  `reset_for_scene` via `is_first` but changes replay recording (reset frame has
  no preceding action). Rejected for blast radius; the callback touches only
  the extractor reset (D1).
- **Growth-curve saturation threshold as the go/no-go** — wrong metric (D3);
  the 1.5 M threshold I used for smoke 5738008 was based on a false assumption
  that a single L1 house saturates well below 1.5 M. It doesn't (a house is ~5 M
  voxels at 0.01 m). Retired.