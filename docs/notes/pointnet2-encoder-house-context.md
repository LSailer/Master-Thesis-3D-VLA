# PointNet++ encoder → live-house VGGT context

How the `external/pointnet2` encoder works, and how to use it as the
**Encoder Module** branch for the `house_context` point cloud produced by
live VGGT.

## 1. What "the encoder" actually is in `external/pointnet2`

`external/pointnet2` is Charles Qi's **original TensorFlow 1.x** implementation.
The classification model (`models/pointnet2_cls_ssg.py`) is:

```
input point cloud  (B, N, 3)
  → SA layer 1   → (B, 512, 128)
  → SA layer 2   → (B, 128, 256)
  → SA layer 3   → (B,   1, 1024)   # group_all → global feature
  → reshape (B, 1024)
  → FC 512 → FC 256 → FC 40         # classification HEAD (not the encoder)
```

**The encoder is the stack of Set Abstraction (SA) layers** (`get_model`
lines 32-34). The three FC layers are a task head; drop them. The
Feature-Propagation (FP) modules in `pointnet_util.py:199` are the *decoder*
(upsampling for segmentation) — not needed for a global context vector.

### The Set Abstraction (SA) module — the core primitive

`pointnet_sa_module` (`utils/pointnet_util.py:87`) does, per layer:

1. **Sample** centroids with farthest-point sampling: `farthest_point_sample`
   → `gather_point` picks `npoint` well-spread centers (`sample_and_group`,
   line 40).
2. **Group** each center's neighborhood: `query_ball_point(radius, nsample)`
   (ball query, capped at `nsample` neighbors) or `knn_point` if `knn=True`
   (line 42-44).
3. **Normalize locally**: subtract the centroid xyz from grouped points
   (`grouped_xyz -= new_xyz`, line 46) → **translation-invariant** local
   patches. If input has features, concat `[feat | rel_xyz]` (`use_xyz`).
4. **Per-point shared MLP**: a stack of `conv2d [1,1]` (= per-point MLP with
   BatchNorm+ReLU) over channels `mlp=[...]` (lines 117-122).
5. **Symmetric pool** over the `nsample` axis: `reduce_max` (line 127) →
   **permutation invariance**. One feature vector per centroid.

Output: `new_xyz (B, npoint, 3)`, `new_points (B, npoint, mlp[-1])`. Stacking
SA layers = a CNN-like hierarchy with a **growing receptive field in metric
space** (radius 0.2 → 0.4 → whole cloud). The last layer uses
`group_all=True` (`sample_and_group_all`, line 59) to pool every point into a
single **1024-d global descriptor**.

### SSG vs MSG

- **SSG** (`pointnet2_cls_ssg.py`): one radius per layer. Simpler, faster.
- **MSG** (`pointnet2_cls_msg.py`): `pointnet_sa_module_msg` runs several
  radii per layer (`[0.1,0.2,0.4]`) and concatenates → robust to **non-uniform
  point density**. VGGT clouds have very uneven density (depth-dependent), so
  MSG is the more principled choice, at higher cost.

### Why it beats the current `house_context` encoder

Today `HousePointsCameraEncoder._house_embedding`
(`src/r2dreamer/encoders/mlp.py:131`) is effectively **PointNet-v1**: a
per-point MLP then a single global mean+max pool. It is permutation-invariant
but has **no spatial locality and no hierarchy** — every point is embedded
independently before one global pool. PointNet++ adds the local grouping +
multi-scale hierarchy, which is exactly what captures room/furniture structure
in a house cloud.

## 2. The blocker: framework mismatch

`external/pointnet2` cannot be dropped into the training path as-is:

- It is **TensorFlow 1.x** (`tf.placeholder`, `tf.variable_scope`), while the
  Encoder Module contract is **Flax `nn.Module` + JAX + bfloat16**
  (`CONTEXT.md`, `CLAUDE.md`).
- Its sampling/grouping/interpolation ops are **hand-written CUDA kernels**
  under `tf_ops/{sampling,grouping,3d_interpolation}` that must be compiled
  against a specific TF ABI. They are not JAX-callable.

So treat `external/pointnet2` as a **reference spec**, not an importable
library. Two viable routes:

| Route | What | Gradients to encoder? | Fit with `dreamer_encoder_setup` |
|---|---|---|---|
| **A. Port SA stack to Flax** (recommended) | Reimplement FPS + ball-query + shared-MLP + max-pool in JAX | Yes — params live in Dreamer's optimized tree | ✅ matches `DreamerEncoderModule` |
| B. Frozen TF extractor | Run TF pointnet2 host-side, emit a fixed vector | No — Dreamer sees a constant | ✗ contradicts README ("encoder updates") |

Route B breaks the explicit decision in
`scratchpad/dreamer_encoder_setup/README.md` ("Why the encoder updates":
encoder params must be inside the differentiated `loss_fn`). Use B only if you
deliberately want a frozen point encoder.

## 3. Mapping live-house VGGT context onto PointNet++ input

Your live context (`README.md` of the setup draft, `HouseContextPoseBuffer`):

```python
encoder_obs["house_context"]  # (4096, 6) bfloat16 = xyz(3) + rgb01(3)
encoder_obs["camera_pose"]    # (9,)  bfloat16
```

PointNet++ split:

```python
l0_xyz    = house_context[..., :3]   # (B, 4096, 3)  geometry → FPS/ball query
l0_points = house_context[..., 3:]   # (B, 4096, 3)  rgb as initial per-point feat
```

RGB rides along as `l0_points`; each SA layer concatenates it with the
locally-normalized `rel_xyz` (`use_xyz=True`) before the shared MLP. The SA
stack replaces `_house_embedding`; the **camera branch and the final
`concatenate([camera_embed, house_embed])` stay exactly as in
`HousePointsCameraEncoder`** so the RSSM interface is unchanged.

### Flax module sketch (Route A)

Illustrative, not production. Uses `RMSNorm` (as elsewhere in `mlp.py`) instead
of BatchNorm.

```python
class HousePointsPointNet2Encoder(nn.Module):
    embed_dim: int = 1024
    camera_hidden: int = 1024
    camera_layers: int = 1
    camera_pose_dim: int = 9

    def _sa(self, xyz, feat, npoint, radius, nsample, mlp, name):
        centers_idx = farthest_point_sample(xyz, npoint)          # (B, npoint)
        new_xyz     = gather(xyz, centers_idx)                    # (B, npoint, 3)
        grp_idx     = query_ball_point(radius, nsample, xyz, new_xyz)
        grp_xyz     = gather(xyz, grp_idx) - new_xyz[:, :, None]  # local norm
        grp_feat    = gather(feat, grp_idx) if feat is not None else grp_xyz
        x = jnp.concatenate([grp_feat, grp_xyz], -1)             # use_xyz
        for i, c in enumerate(mlp):                               # shared MLP
            x = nn.Dense(c, name=f"{name}_mlp{i}")(x)
            x = RMSNorm(name=f"{name}_norm{i}")(x); x = nn.silu(x)
        return new_xyz, jnp.max(x, axis=2)                        # symmetric pool

    @nn.compact
    def __call__(self, obs):
        cp, lead = flatten_event(obs[CAMERA_POSE_KEY], event_ndims=1)
        cam = self._camera_embedding(cp)                          # reuse today's branch

        hc  = jnp.asarray(obs[HOUSE_CONTEXT_KEY])                 # (S, 4096, 6)
        if hc.ndim == 2: hc = hc[None]
        xyz, feat = hc[..., :3], hc[..., 3:]
        xyz, feat = self._sa(xyz, feat, 512, 0.2, 32, [64,64,128],  "sa1")
        xyz, feat = self._sa(xyz, feat, 128, 0.4, 64, [128,128,256],"sa2")
        glob = jnp.max(nn.Dense(1024, name="sa3")(feat), axis=1)  # group_all
        house = nn.Dense(self.embed_dim, name="house_proj")(glob)
        # broadcast singleton scene cloud across camera batch (as today)
        house = broadcast_house(house, cam.shape[0])
        return jnp.concatenate([restore_leading(cam, lead),
                                restore_leading(house, lead)], axis=-1)
```

You need JAX versions of `farthest_point_sample`, `query_ball_point`, `gather`
(there is no CUDA dependency — plain `jnp` + `jax.lax.top_k`/`argsort` suffice
at N=4096). kNN grouping (`knn_point`) is often easier and jit-friendlier than
ball query in pure JAX: compute pairwise dists, take `top_k` nearest.

## 4. Caveats specific to live VGGT house context

1. **Metric scale / normalization.** PointNet++ radii (0.2, 0.4) are tuned for
   ModelNet clouds normalized to the **unit sphere**. VGGT world points are in
   VGGT's arbitrary (up-to-scale) frame. Either (a) normalize each scene cloud
   to unit sphere before the encoder, or (b) set radii to your Habitat metric
   scale (buffer `voxel_size_m=0.05` m → room-scale radii ~0.2–0.8 m). Without
   this, ball query returns garbage neighborhoods. **Normalize consistently
   across scenes** or the encoder won't generalize.
2. **Padding.** The buffer pads to `max_points=4096`. Zero/pad points corrupt
   FPS and ball query (they get sampled as real centroids). Carry a validity
   mask and either exclude padded points from FPS or set their coords to `+inf`
   distance. The original TF code assumes all points valid.
3. **Cost vs. the singleton broadcast.** `house_context` is a **static
   per-scene sidecar** broadcast across the `(B*T)` camera poses
   (`_house_embedding` lines 148-149). Keep that: run the (expensive) SA stack
   **once per scene**, then broadcast — do **not** let it run per replay
   timestep. This is what makes a heavy PointNet++ affordable inside
   `train_step`.
4. **No BatchNorm.** The TF model leans on BN with `bn_decay`/`is_training`.
   BN is fragile across Dreamer's sequence batches and the singleton house
   batch (batch size 1 per scene → BN stats undefined). Use RMSNorm/LayerNorm,
   matching `mlp.py`.
5. **MSG cost.** MSG (multi-radius) is more faithful to non-uniform VGGT
   density but multiplies grouping work per layer. Start SSG; move to MSG only
   if the global descriptor underfits scene structure.

## 5. Concrete integration steps

1. Add `HousePointsPointNet2Encoder` next to `HousePointsCameraEncoder` in
   `src/r2dreamer/encoders/mlp.py` (or a new `encoders/pointnet2.py`).
2. Implement JAX `farthest_point_sample` + kNN/ball grouping helpers (no CUDA).
3. Register it in the encoder-construction path the same way
   `HousePointsCameraEncoder` is selected; keep the `__call__(obs) → (…,
   embed_dim)` contract from `DreamerEncoderModule`.
4. Normalize the house cloud in **Observation Preparation**
   (`live_vggt`/`HouseContextPoseBuffer`), not in the encoder, and add the
   padding mask to `house_context`.
5. Verify: unit test permutation invariance (shuffle the 4096 points → same
   embedding within tolerance) and singleton-broadcast shape `(B*T, embed_dim)`
   before wiring into training.

## References in repo
- `external/pointnet2/utils/pointnet_util.py:22-197` — sample/group + SA modules
- `external/pointnet2/models/pointnet2_cls_ssg.py:20-44` — SSG encoder stack
- `external/pointnet2/models/pointnet2_cls_msg.py:17-39` — MSG encoder stack
- `src/r2dreamer/encoders/mlp.py:102-179` — current house/camera encoder to mirror
- `scratchpad/dreamer_encoder_setup/README.md` — Dreamer-owned encoder contract
