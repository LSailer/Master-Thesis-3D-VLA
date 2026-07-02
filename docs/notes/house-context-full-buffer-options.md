# Feeding more of the house buffer to the encoder

Why the encoder snapshot is capped at `HOUSE_CONTEXT_MAX_POINTS = 4096`, and
concrete options for using more (or effectively all) of the buffer. Companion
to [live-house-context-pipeline.md] and [point-dedup-literature.md].

## Why 4096 today

1. **Static shapes.** The encoder input shape is baked into the compiled train
   graph; any growth recompiles everything.
2. **Dense cost on padding.** XLA pays full per-row MLP cost on padded rows.
   At buffer capacity (2²³ rows): first Dense layer ≈ 8.6 GB activations
   (8.4M × 256 f32) and ~10¹¹ FLOPs per train step — for ~2.5% real rows at a
   typical 210k-point map.
3. **Unmasked pooling breaks under padding.** `mean` over mostly-zero rows
   dilutes the real signal by the padding ratio and makes embedding magnitude
   a function of fill level; `max` clips negative feature channels at the
   zero-row value.

The resample-to-4096 snapshot (`_house_context_snapshot`) avoids all three by
repeating/striding *real* points — but discards ~98% of a 210k-point map per
step.

## Option 1 — masked pooling + bigger fixed budget (do this first)

Keep a single static shape `(N, 6)` with `N` = 32k–64k, zero-pad, and pass the
true count. Masking is *data*, not shape → no recompiles.

```python
# adapter: snapshot = first-size rows zero-padded to N; also emit size
# encoder:
mask = jnp.arange(N)[None, :, None] < size            # (1, N, 1) bool
x = per_point_mlp(points)                             # zeros still computed
mean = (x * mask).sum(1) / jnp.maximum(size, 1)       # masked mean
maxp = jnp.where(mask, x, -jnp.inf).max(1)            # masked max
```

Costs: the house branch runs **once per batch** (singleton broadcast), so even
N = 262k is plausibly a few ms — bench before assuming it's too big. Wasted
FLOPs on padding are bounded by choosing N near the realistic map size
(1–2M voxels max for a house at 1 cm; 256k at 5 cm-equivalent sampling), not
at worst-case capacity.

Change surface: adapter emits `(snapshot, size)`; `HousePointsCameraEncoder`
gets ~6 new lines; observation contract gains one scalar field.

## Option 2 — spatially uniform snapshot (composable with 1)

Today's stride over insertion order is uniform over *discovery time*, not
space. Cheap fix that reuses existing machinery: keep a second, coarser
"summary" voxel assignment (e.g. 10 cm) and snapshot one representative per
coarse voxel. Bounded count (a house has ~50–100k occupied 10 cm voxels),
spatially even, and it's the same sort/dedup graph the buffer already runs.

## Option 3 — bucketed shapes

Compile the encoder at a small ladder of sizes (4k / 32k / 256k / 2M) and pick
the bucket by current fill. Standard JAX padding-bucket pattern: recompiles are
bounded by the ladder length (each ~1–2 s, once per run). Downsides: code
complexity in the trainer (obs shape changes at bucket boundaries also touch
the replay/batch plumbing), and the world-model graph downstream of the
encoder must either re-specialize too or meet at a fixed embedding — fine
here since the encoder output is always 2048.

## Option 4 — scatter-pooled voxel feature grid (best long-term)

Move the summarization *into the buffer's JIT graph* instead of the encoder:

1. Maintain, alongside `store_xyz/rgb`, a fixed coarse grid, e.g. 32³ cells
   over the scene bounds (or a hash of active blocks), each holding
   scatter-mean of point features + an occupancy count channel.
2. Update incrementally in `add` via `.at[cell].add(...)` — same scatter
   machinery the hash insert already uses, negligible cost.
3. The encoder consumes the fixed `(32³, C)` grid (as tokens or a 3D conv
   volume). Buffer size becomes irrelevant to encoder cost forever; empty
   cells are explicit (count = 0) rather than fake zero-points.

This is the per-block-payload idea from VDB / Nießner voxel hashing applied to
features, and it dovetails with the PointNet++ prototype: set-abstraction
levels are essentially learned versions of this grouping.

## Option 5 — latent cross-attention (Perceiver-style)

Learned latent queries (e.g. 64) cross-attend over point tokens with a key
padding mask; cost is `N × latents`, linear in N, and masking is native to
attention. Biggest modelling upgrade, biggest implementation cost; consider
only after the PointNet++ variant is evaluated, since both compete for the
same "local structure" budget.

## Recommendation

1. Option 1 now (masked pooling, N = 32k–65k, bench the house-branch ms).
2. Option 2 alongside it for representativeness.
3. Option 4 when/if encoder cost or map fidelity becomes the binding
   constraint; it subsumes the snapshot concept entirely.
