## 1. Precondition — make the buffer path runnable

- [x] 1.1 Inspect the working-tree diff of `src/buffer/house_context_pose_buffer.py`; fix or revert the in-progress refactor so `_add_frame_to_state` has a defined `flat_rgb` and integer voxel keys
- [x] 1.2 Verify `HouseContextPoseBuffer.add()` traces under JIT and `save()` writes a valid colored PLY (reuse `scripts/r2dreamer/check_episode_frame_alignment.py` or a minimal repro)

## 2. Token-half exposure plumbing

- [x] 2.1 In `src/vggt/jax/feature_extractor.py`, expose the frame-half `[:1024]` and global-half `[1024:]` for each DPT-consumed layer (indices 4, 11, 17, 23), not only the final layer (extend the split at `feature_extractor.py:944-947`)
- [x] 2.2 Return the halves per-patch `(1369, 1024)` for a single frame; add a shape guard that rejects a non-`(1369,1024)` (e.g. pooled `(1,1024)`) half
- [x] 2.3 Confirm the four exposed layer indices match `dpt_head.py:391` `intermediate_layer_idx`

## 3. Point-head re-entry

- [x] 3.1 Add a re-entry (near `_point_head_apply` / `_run_heads`, `feature_extractor.py:781-804`) that accepts an externally-assembled `aggregated_tokens_list` and returns `world_points` + confidence without running the aggregator
- [x] 3.2 Enforce 2048-channel width per consumed layer before invoking the head; raise on mismatch
- [x] 3.3 Unit-check: feeding a re-run's own `out_list` through the re-entry reproduces the normal reconstruction bit-for-bit

## 4. Unified colored-PLY writer

- [x] 4.1 Choose the 6-property XYZRGB writer (`feature_extractor.py:955` `write_point_cloud_ply`) as the single export path for all arms (D3)
- [x] 4.2 Route point positions from `world_points` and RGB from the configured color-source frame's pixels; record which frame supplied color
- [x] 4.3 Verify each emitted PLY round-trips via `static_house_context.load_ascii_ply_xyzrgb`

## 5. Arm drivers

- [x] 5.1 Arm A (baseline): `ResetMode.FULL` fresh cache, single Image 1 → point head → colored PLY
- [x] 5.2 Arm B (house context): stream 500 same-house frames in a fixed, recorded order under `ResetMode.PERSIST_SCENE`, then Image 1 → point head → colored PLY
- [x] 5.3 Arm C (token surgery): get global half from the 500-frame pass + frame half from Image 1's pass; assemble `concat(frame_half, global_half)` per consumed layer; call the re-entry → colored PLY, colored by Image 1
- [x] 5.4 Add a flag on Arm C to splice all four consumed layers (default) vs final-layer-only (secondary comparison)
- [x] 5.5 Fix seed + frame order so Arm B/Arm C are reproducible; verify two identical runs match

## 6. Comparison & verification

- [x] 6.1 Run all three arms on one target frame and export three structurally identical PLYs
- [x] 6.2 Confirm all three PLYs share the same header/property set and load in CloudCompare/Blender
- [x] 6.3 (Optional) log a quantitative delta across arms (point count, confidence stats, Chamfer) to accompany the visual comparison
- [x] 6.4 `openspec validate global-token-reconstruction-ablation --strict`
