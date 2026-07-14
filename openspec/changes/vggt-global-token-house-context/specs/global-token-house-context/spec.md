## ADDED Requirements

### Requirement: House context is derived from Aggregator global patch tokens

The system SHALL derive the live house-context signal from VGGT's Aggregator
**global tokens** (the global half of the `(1374, 1024)` token set), and SHALL
select **only the 1369 patch tokens**, discarding the 4 register tokens and the 1
camera token. The point-cloud path (`world_points` → `HouseContextPoseBuffer`)
SHALL NOT be required for this signal.

#### Scenario: Patch tokens selected from global tokens

- **WHEN** the encoder receives the Aggregator global tokens of shape `(…, 1374, 1024)`
- **THEN** it uses exactly the `(…, 1369, 1024)` patch tokens as its input
- **AND** the camera token and the 4 register tokens do not contribute to the house-context embedding

#### Scenario: No point-cloud dependency

- **WHEN** the house-context signal is produced for a frame
- **THEN** no voxel buffer or point-cloud snapshot is consulted to produce it

### Requirement: Patch tokens are reduced to a single (1, 1024) house embedding by a permutation-invariant PointNet reducer

The reducer SHALL apply a shared per-token MLP (`Dense → RMSNorm → SiLU`, identical
weights for every token, no token-to-token interaction), then a **single max-pool**
over the token axis, then a `Dense(1024)` projection, producing a `(…, 1024)` house
embedding. The reduction SHALL be permutation-invariant over the 1369 tokens and
SHALL NOT include a mean-pool branch, a flatten-then-Dense reduction, or a camera
side branch.

#### Scenario: House-branch shape and width

- **WHEN** patch tokens `(…, 1369, 1024)` are reduced
- **THEN** the house embedding has feature width `1024` with leading dims preserved (`(…, 1024)`)
- **AND** no camera-token side branch contributes to it (the RGB fusion is a separate requirement)

#### Scenario: Permutation invariance

- **WHEN** the 1369 patch tokens are presented in any order
- **THEN** the reduced `(…, 1024)` house embedding is unchanged

### Requirement: The encoder fuses the house embedding with an RGB conv embedding (hybrid)

The encoder SHALL encode the per-step RGB image (`hybrid_image`, `(…, 3, 64, 64)`)
through a `ConvEncoder` branch projected to width `1024` (`embed_dim=1024`), and
SHALL concatenate that RGB embedding with the `(…, 1024)` PointNet house embedding,
producing a single fused observation embedding of width `2048`. The RGB conv branch
and the patch-token PointNet branch SHALL be the only two branches; no camera-token
branch SHALL contribute.

#### Scenario: RGB and tokens both encoded and fused

- **WHEN** a frame provides `hybrid_image` `(…, 3, 64, 64)` and `global_patch_tokens` `(…, 1369, 1024)`
- **THEN** the ConvEncoder produces a `(…, 1024)` RGB embedding and the PointNet reducer produces a `(…, 1024)` house embedding
- **AND** the encoder returns their concatenation `(…, 2048)`

#### Scenario: Only two branches contribute

- **WHEN** the fused embedding is built
- **THEN** the contributing branches are exactly the RGB conv and the patch-token PointNet
- **AND** the camera token does not contribute

### Requirement: Cross-episode scene memory uses PERSIST_SCENE with DPT heads off

The system SHALL run the VGGT extractor in `ResetMode.PERSIST_SCENE`, saving and
restoring the Aggregator KV cache keyed by `scene_id` across episodes of the same
house, and SHALL run with the DPT point head disabled (heads-off), since the token
path does not consume `world_points`.

#### Scenario: Cache persists across episodes of the same scene

- **WHEN** a new episode begins in a scene whose `scene_id` was seen before
- **THEN** the Aggregator KV cache for that `scene_id` is restored rather than wiped

#### Scenario: Point head not computed

- **WHEN** a frame is processed on the global-token house-context path
- **THEN** the DPT point head is not evaluated and no point-head cache is allocated

### Requirement: The fused embedding integrates with the world model unchanged

The `(…, 2048)` fused observation embedding (RGB conv ⊕ PointNet house) SHALL be
provided to the world model as an observation embedding, concatenated with the other
observation embeddings and consumed by `R2RSSM` without any change to the RSSM's
deterministic size, stochastic size, posterior head, or prior head.

#### Scenario: RSSM contract unchanged

- **WHEN** the fused embedding is fed to the world model during training or acting
- **THEN** it is consumed as a standard observation embedding
- **AND** the RSSM's `deter` (2048), `stoch` (32×16), and `feat` (2560) dimensions are unchanged
