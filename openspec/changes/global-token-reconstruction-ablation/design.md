## Context

VGGT's aggregator produces, per token, a 2048-wide vector that is the concatenation of a per-frame ("frame half", channels `[:1024]`) and a cross-frame ("global half", channels `[1024:]`) stream (`aggregator.py:507`; split read at `feature_extractor.py:944-947`). The two halves are one coupled stream — the global half feeds the next layer, and the frame half is a per-layer snapshot — so "frame vs global" is a **channel split**, not two runnable sub-networks. There is no single pooled "global token": the global half is per-patch (1369 patch tokens × 1024) for a single frame.

The DPT point head (`dpt_head.py:456`) consumes the concatenated 2048-wide patch tokens from **four** layers (indices 4, 11, 17, 23) and outputs `world_points` + confidence; color is not a head output — RGB is the source image pixel reprojected 1:1 (`_last_rgb`). Streaming `extract()` already accumulates context frame-by-frame (`benchmark_streaming.py:67`); the aggregator KV cache is self-bounding but its eviction is lossy and order-dependent. `ResetMode.FULL` vs `PERSIST_SCENE` (`feature_extractor.py:611`) give the no-context vs context switch.

This change is heads-ON and produces colored PLYs; the sibling change `vggt-global-token-house-context` is heads-OFF and produces an RSSM embedding. They share token-structure knowledge but no code path.

## Goals / Non-Goals

**Goals:**
- Isolate the global half's contribution to a target frame's reconstruction via three comparable colored-PLY arms (baseline / house-context / token-surgery).
- Add minimal, additive plumbing: expose the four consumed layers' halves, and a point-head re-entry accepting an externally-assembled token list.
- Make all three arms emit structurally identical PLYs for fair comparison.
- Keep Arm B/Arm C reproducible by fixing frame order.

**Non-Goals:**
- No change to the aggregator, KV-cache eviction, or attention internals.
- No pooled "scene vector" global token (that is the sibling change's PointNet reducer, explicitly out of scope here).
- No modification to the RSSM embedding path or `vggt-global-token-house-context`'s requirements.
- No new learned components — this is analysis/interpretability, not training.

## Decisions

**D1 — Arm C is a per-patch half-splice across the 4 consumed layers, not a pooled vector.**
The head reads layers 4/11/17/23, each `(B, S, 1369, 2048)`. Arm C assembles, for each of those layers, `concat(frame_half_targetPass, global_half_500Pass)` at channel 1024, forming a synthetic `aggregated_tokens_list`, then calls the point head. Alternative (a single pooled global vector broadcast into patches) was rejected: it does not fit the per-patch slot and would require inventing a tiling scheme the head does not support — and it would no longer be VGGT's actual global token.

**D2 — Additive plumbing, no head signature change.** The head already accepts a 2048-wide token list (`dpt_head.py:460`). We add (i) exposure of the four consumed layers' halves (today only the final layer is surfaced) and (ii) a re-entry wrapper that takes an externally-assembled list and bypasses `_run_aggregator` (near `_point_head_apply` / `_run_heads`, `feature_extractor.py:781-804`). Alternative (recompute the aggregator inside Arm C) was rejected — it cannot mix halves from two different passes.

**D3 — Unify on one PLY writer.** Two writers exist with different headers: the 9-property CloudCompare writer `HouseContextPoseBuffer.save()` (`house_context_pose_buffer.py:511`) and the 6-property `write_point_cloud_ply` (`feature_extractor.py:955`). Pick the 6-property XYZRGB writer as the common path for all arms (it already round-trips with `static_house_context.load_ascii_ply_xyzrgb`); Arm B reads points from its reconstruction, not from the voxel buffer, so it does not need the buffer's CloudCompare scalar fields. Comparability across arms outweighs CloudCompare scalar convenience.

**D4 — Context via streaming `extract()` + `ResetMode`.** Arm A = `FULL` + single frame; Arm B/C = `PERSIST_SCENE` warmed by 500 frames in fixed order. Reuses existing streaming loop; no new accumulation machinery.

**D5 — Color source is explicit and defaults to the target frame.** Since color comes from `_last_rgb`, Arm C (whose tokens mix two passes) colors with Image 1 by default and records the choice, so cross-arm color is identical and only geometry differs.

## Risks / Trade-offs

- **Lossy, order-dependent KV eviction** → after 500 frames the cache is a similarity-pruned subset, so "global half after 500 frames" is conditioned on an evicted cache. Mitigation: fix and record frame order + seed; optionally log per-block `valid_len` / `last_scores` to characterize what survived.
- **Only the final-layer halves are exposed today** → Arm C needs 4 layers. Mitigation: D2 exposure plumbing; verify the four layer indices match `dpt_head.py:391`.
- **Broken buffer refactor blocks Arm B** → `house_context_pose_buffer.py` working tree has `NameError` on `flat_rgb` and non-int voxel keys. Mitigation: precondition task to fix or revert before Arm B; Arm A/C reconstruct via the point head and do not depend on the buffer.
- **Splice semantics ambiguity** → swapping the global half at only the final layer vs all four consumed layers gives different results. Mitigation: default to all four (faithful); expose a flag to ablate final-only as a secondary comparison.
- **Camera-head cache growth under PERSIST_SCENE** → only matters if camera head is on; the point-only reconstruction can keep the camera head off to avoid the sliding-window concat cost.

## Migration Plan

Analysis-only, no production surface. Order: (1) fix/revert the buffer refactor; (2) add consumed-layer half exposure; (3) add point-head re-entry; (4) unify PLY writer; (5) implement Arm A, then B, then C drivers; (6) run all three on one target frame + compare. Rollback = drop the driver scripts and the additive plumbing; no existing behavior changes.

## Open Questions

- Splice scope for Arm C: all four consumed layers (default) vs final layer only — keep both as comparison points, or commit to one?
- Should Arm C also expose the reverse splice (global half from Image 1 + global-context frame half) as a symmetric control?
- Which 500-frame ordering best represents "house context" — trajectory order, or a fixed shuffle? (Cross-reference `vggt-global-token-house-context` risk on PERSIST_SCENE fidelity.)
- Do we quantify reconstruction difference (e.g. Chamfer / point count / confidence stats) in this change, or only produce the PLYs for visual comparison?
