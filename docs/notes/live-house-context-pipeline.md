# Live house context: buffer → snapshot → encoder

How the `vggt_house_points_pose` encoder (run-id
`habitat-l1-vggt-house-points-pose`) consumes the live per-scene point buffer,
and what actually reaches the network — including when zeros appear and when
they don't.

## Data flow per env step

```
env_obs (518² RGB)
  └─ JAXVGGTFeatureExtractor.extract          # VGGT forward + DPT head, ~64 ms
       ├─ camera_pose (9,)                    ──────────────┐
       └─ world_points (518,518,3) + confidence (518,518)   │
            └─ HouseContextPoseBuffer.add     # voxel dedup, ~2 ms
                 └─ house_context_array(262_144)  # (snapshot, size) tuple
                      ├─ agent_obs[HOUSE_CONTEXT_KEY]      ─┤ (262_144, 6)
                      └─ agent_obs[HOUSE_CONTEXT_SIZE_KEY] ─┤ scalar int32
                                                            ▼
                                     HousePointsCameraEncoder (encoders/mlp.py)
                                       camera branch: MLP(9) → 1024
                                       house branch : per-point MLP(6→256→256)
                                                      → masked [mean ‖ max] → 1024
                                       output: concat → 2048
```

Wiring: `VGGTHousePointsPoseEncoder` (encoders/house_points_pose.py) declares
`module_cls = HousePointsCameraEncoder` and builds
`VGGTHousePointsPoseObsAdapter` (adapters/hybrid_adapter.py), which owns one
`HouseContextPoseBuffer` per `scene_id` (buffers persist across episodes of the
same scene; the VGGT KV-cache is reset per episode).

## The snapshot: zero-padding plus a true-size scalar

`buffer.house_context_array(max_points)` calls `_house_context_snapshot`
(house_context_pose_buffer.py), which returns a **tuple**
`(snapshot, size)` — the `(262_144, 6)` tensor and the number of valid rows:

- **size ≤ 262_144** (the expected case for a whole house at 1 cm voxels,
  ~210k after 50 steps on one L1 scene): the stored prefix is copied verbatim
  and rows past `size` are **zeros**. No repetition, no information loss —
  every unique point the agent has ever seen reaches the encoder.
- **size > 262_144** (overflow fallback only): an even stride over insertion
  order, same behaviour as the old resampling design.
- **size == 0** (only before the very first `add` of a scene): all-zero
  snapshot with `size = 0`; `_empty_house_context` provides the same tensor
  before the adapter has seen any frame.

The padding zeros are *not* fed to the pooling: the size scalar travels with
the snapshot as `HOUSE_CONTEXT_SIZE_KEY` (obs key `"house_context_size"`)
through `transform()` (live path) and `augment_replay_batch` (replay path),
and the encoder masks on it.

## How the encoder treats the points (and the zero edge case)

`HousePointsCameraEncoder._house_embedding` (encoders/mlp.py:131) is a
PointNet-style set encoder:

1. A shared 2-layer MLP (Dense 256 + RMSNorm + SiLU) is applied to every row
   `(x, y, z, r, g, b)` independently — no point interacts with any other.
2. **Masked** symmetric pooling collapses the 262_144 rows using the size
   scalar: mean = sum over valid rows / `max(size, 1)`; max = max over rows
   with invalid rows set to `-inf` (and zeroed if `size == 0`). Padding rows
   therefore contribute nothing to either statistic. If no size key is present
   (legacy paths), pooling falls back to unmasked mean/max over all rows.

Consequences:

- **Padding zeros are invisible.** The MLP still runs on padded rows (wasted
  FLOPs are cheap: full-N house branch is ~0.9 ms fwd / ~2.3 ms fwd+bwd on
  H100, job 5695204), but the mask keeps them out of the pooled embedding, so
  the embedding is exactly what a variable-length encoder would produce.
- **The all-zero frame is harmless.** With `size == 0` the guard emits a zero
  max and a zero mean — a distinguishable "no map yet" token rather than
  garbage. It occurs for at most the first step of a fresh scene.
- **Permutation invariance** comes for free from the pooling, which is why the
  arbitrary insertion ordering of the snapshot doesn't matter.

Batch mechanics: the house branch is pooled once from the *latest* snapshot
`(1, 262_144, 6)` and broadcast across the whole `(B·T)` camera batch
(`augment_replay_batch` injects it into sampled replay batches). This is exact
for the single-house L1 curriculum and an acknowledged approximation for
multi-scene training (docstring in hybrid_adapter.py).

## Which variant, and what's next

- **In use:** `HousePointsCameraEncoder` — flat per-point MLP + mean/max pool
  (PointNet-lite: no T-Net, no local neighborhoods, no hierarchy).
- **In progress (uncommitted):** `PointNet2FeatureEncoder` /
  `PointNet2Encoder` (encoders/pointnet2.py) — PointNet++ set-abstraction
  levels (SSG/MSG) that build local geometric features by grouping neighbors
  before pooling. See [pointnet2-encoder-house-context.md] for that design.
  The flat encoder can only capture global statistics of the cloud; the
  PointNet++ variant is the planned upgrade for actual local structure.

## Numbers worth remembering

| quantity | value | where |
|---|---|---|
| snapshot shape | (262_144, 6) float16, zero-padded | `HOUSE_CONTEXT_MAX_POINTS`, encoders/constants.py |
| valid-row count | scalar int32 | `HOUSE_CONTEXT_SIZE_KEY`, observation_keys.py |
| point features | xyz + rgb in [0, 1] | `HOUSE_POINT_DIM` |
| buffer capacity | 2²³ ≈ 8.4M voxels @ 1 cm | `BUFFER_CAPACITY`, hybrid_adapter.py |
| house embedding | 1024 (pooled), concat w/ 1024 camera → 2048 | encoders/mlp.py |
| house branch cost at full N | ~0.9 ms fwd / ~2.3 ms fwd+bwd (H100) | bench job 5695204 |
