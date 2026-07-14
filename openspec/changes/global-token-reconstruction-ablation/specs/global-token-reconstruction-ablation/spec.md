## ADDED Requirements

### Requirement: Heads-ON colored point-cloud reconstruction

The system SHALL run the VGGT DPT point head to reconstruct a colored point cloud for a target frame. Point positions SHALL come from the point head's `world_points`; per-point RGB SHALL come from the source image pixel reprojected 1:1 by grid position (the head does not emit color). The reconstruction path SHALL be selectable independently of the heads-OFF embedding path used by `vggt-global-token-house-context`.

#### Scenario: Single frame reconstructs to a colored cloud

- **WHEN** a target frame is passed through the aggregator and DPT point head with heads ON
- **THEN** the system produces `world_points` of shape `(H, W, 3)` paired with RGB `(H, W, 3)` uint8 taken from that frame's pixels
- **AND** flattens them to an XYZRGB point set ready for PLY export

#### Scenario: Color source is explicit

- **WHEN** the frame providing pixels for color differs from a frame providing tokens
- **THEN** the system SHALL color the cloud using the configured color-source frame (default: the target frame, Image 1) and record which frame supplied the color

### Requirement: Three-arm reconstruction protocol

The system SHALL provide three reconstruction arms over the same target frame, each emitting a colored PLY: Arm A (baseline, no context), Arm B (house context via warmed KV cache), and Arm C (token surgery). The 500-frame context used by Arm B and Arm C SHALL be applied in a fixed, recorded order because KV-cache eviction is lossy and order-dependent.

#### Scenario: Arm A — baseline, no context

- **WHEN** the extractor is reset to a fresh cache (`ResetMode.FULL`, KV-cache = 0) and only the target frame is processed
- **THEN** the system emits a colored PLY reconstructed from the target frame alone with no prior context

#### Scenario: Arm B — house context via warmed cache

- **WHEN** 500 prior same-house frames are streamed in a fixed order under `ResetMode.PERSIST_SCENE`, then the target frame is processed
- **THEN** the target frame's reconstruction attends over the accumulated (evicted) KV cache
- **AND** the system emits a colored PLY reconstructed with house context

#### Scenario: Arm C — token surgery

- **WHEN** the global half is taken from the 500-frame pass and the frame half from the target frame's pass, spliced and fed to the point head
- **THEN** the system emits a colored PLY reconstructed from the recombined tokens

#### Scenario: Context order is reproducible

- **WHEN** Arm B or Arm C is run twice with the same fixed frame order and seed
- **THEN** the two runs produce identical warmed-cache state and identical reconstructions

### Requirement: Per-patch frame/global half exposure for consumed layers

The system SHALL expose VGGT's frame-half and global-half channel slices (`[:1024]` and `[1024:]` of the 2048-wide tokens) for each DPT-consumed layer (indices 4, 11, 17, 23), not only the final layer. Each exposed half SHALL retain the per-patch layout (1369 patch tokens × 1024 channels) for a single frame. The system SHALL NOT treat the global half as a single pooled scene vector.

#### Scenario: Consumed-layer halves are available

- **WHEN** a frame is processed and the consumed-layer halves are requested
- **THEN** the system returns, for each of layers 4/11/17/23, a frame-half and a global-half of shape `(1369, 1024)` for that frame

#### Scenario: Reject pooled-vector misuse

- **WHEN** a caller supplies a global half that is not per-patch `(1369, 1024)` (e.g. a single pooled `(1, 1024)` vector)
- **THEN** the system SHALL raise a shape error rather than broadcast it into the patch slot

### Requirement: Point-head re-entry with externally-assembled tokens

The system SHALL provide a point-head re-entry that accepts an externally-assembled `aggregated_tokens_list` (2048-wide tokens for the consumed layers) and returns `world_points` + confidence, without re-running the aggregator. Arm C SHALL assemble this list by concatenating a frame half and a global half at channel 1024 for each consumed layer.

#### Scenario: Synthetic token list drives the head

- **WHEN** a synthetic `aggregated_tokens_list` is built as `concat(frame_half, global_half)` (axis = channels) for each consumed layer and passed to the re-entry
- **THEN** the point head produces `world_points` + confidence without invoking the aggregator

#### Scenario: Channel width is enforced

- **WHEN** an assembled token for any consumed layer is not exactly 2048 channels wide
- **THEN** the system SHALL raise an error before invoking the head

### Requirement: Unified colored-PLY export

All three arms SHALL emit colored PLYs through a single writer so the outputs are structurally comparable (same header and property set). The chosen writer SHALL round-trip with the existing PLY reader used by the pipeline.

#### Scenario: Structurally comparable outputs

- **WHEN** Arm A, Arm B, and Arm C each export a PLY for the same target frame
- **THEN** all three files use the same PLY header and property set
- **AND** each loads back via the pipeline's PLY reader without error

### Requirement: Arm B precondition — buffer refactor is runnable

Before Arm B is exercised, the accumulation path in `src/buffer/house_context_pose_buffer.py` SHALL trace and run. The in-progress working-tree refactor (undefined `flat_rgb`, non-integer voxel keys) SHALL be fixed or reverted.

#### Scenario: Accumulation path traces

- **WHEN** frames are added to `HouseContextPoseBuffer` and the state is saved
- **THEN** `add()` / `_add_frame_to_state` trace under JIT without `NameError` or dtype errors
- **AND** `save()` writes a valid colored PLY
