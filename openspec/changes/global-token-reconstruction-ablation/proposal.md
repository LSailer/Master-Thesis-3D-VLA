## Why

We want to know how much of VGGT's reconstruction quality on a target frame comes from the target image itself versus the accumulated house context carried in the aggregator's global attention. VGGT couples a per-frame ("frame half") and a cross-frame ("global half") channel stream in every token; there is currently no way to isolate their individual contribution to the point map. A controlled 3-arm ablation over colored point clouds makes the global token's effect directly visible and measurable.

This is deliberately separate from `vggt-global-token-house-context`: that change runs the DPT heads OFF and reduces the global patch tokens to a `(1,1024)` embedding for the RSSM. This change runs the DPT point head ON and reconstructs colored PLYs. It cross-references that change's open question "Does PERSIST_SCENE measurably beat FULL?", which is exactly Arm A vs Arm B here.

## What Changes

- Add a 3-arm reconstruction ablation, each arm producing a colored PLY of the same target frame (Image 1):
  - **Arm A — baseline**: `ResetMode.FULL` / fresh cache, single Image 1 → VGGT → point head → colored PLY (KV-cache = 0, no context).
  - **Arm B — house context**: stream 500 prior same-house frames (fixed order) to warm the aggregator KV cache under `ResetMode.PERSIST_SCENE`, then Image 1 → point head → colored PLY.
  - **Arm C — token surgery (per-patch half-splice)**: obtain the GLOBAL half from the 500-frame pass and the FRAME half from Image 1's pass, splice `concat(frame_half_imageB, global_half_run500)` at channel 1024 for **each** of the 4 DPT-consumed layers (indices 4, 11, 17, 23), feed the synthetic `aggregated_tokens_list` to the point head → colored PLY.
- **NEW plumbing**: expose the frame/global channel halves for the 4 DPT-consumed layers (today only the final layer's halves are surfaced on `VGGTExtractOutput`), and add a point-head re-entry that accepts an externally-assembled `aggregated_tokens_list`.
- **Unify PLY output**: adopt a single colored-PLY writer so all three arms emit structurally comparable files (today two writers with different headers exist).
- **Precondition fix**: repair or revert the broken in-progress refactor in `src/buffer/house_context_pose_buffer.py` that currently breaks the Arm B accumulation path.
- Fix frame order for the 500-frame context so Arm B / Arm C are reproducible (KV-cache eviction is lossy and order-dependent).

## Capabilities

### New Capabilities
- `global-token-reconstruction-ablation`: a heads-ON, 3-arm reconstruction protocol (baseline / house-context / token-surgery) that isolates the contribution of VGGT's global channel-half to the reconstructed colored point cloud, including the token-half exposure + point-head re-entry plumbing and a unified colored-PLY export it depends on.

### Modified Capabilities
<!-- None. `openspec/specs/` is currently empty; the related change `vggt-global-token-house-context` is only cross-referenced, its requirements are not modified here. -->

## Impact

- **Read/plumbing (new)** in `src/vggt/jax/feature_extractor.py`: expose the frame/global halves for layers 4/11/17/23 (extend beyond the final-layer split at `feature_extractor.py:944-947`); add a point-head re-entry that takes an externally-assembled `aggregated_tokens_list` (around `_point_head_apply` / `_run_heads`, `feature_extractor.py:781-804`).
- **Point head** `src/vggt/jax/heads/dpt_head.py:456`: no signature change — it already accepts a 2048-wide `aggregated_tokens_list`; Arm C supplies a synthetic one.
- **PLY export**: unify on one writer — `HouseContextPoseBuffer.save()` (9-property CloudCompare header, `house_context_pose_buffer.py:511`) vs `write_point_cloud_ply` (6-property, `feature_extractor.py:955`).
- **Precondition**: `src/buffer/house_context_pose_buffer.py` working-tree refactor (undefined `flat_rgb`, non-int voxel keys) must be fixed/reverted before Arm B runs.
- **Context accumulation**: reuses streaming `extract()` (`benchmark_streaming.py:67`) and `ResetMode.FULL` / `PERSIST_SCENE`; no change to the aggregator or KV-cache internals.
- **Color**: RGB is taken from the source image pixel reprojected 1:1 (`_last_rgb`), not produced by the head; Arm C is colored by Image 1 by default.
- Cross-references change `vggt-global-token-house-context` (heads-OFF RSSM embedding) — no overlap in code paths.
