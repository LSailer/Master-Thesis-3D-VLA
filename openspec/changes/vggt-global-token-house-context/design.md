## Context

VGGT's Aggregator ("alternating frame/global attention tower",
`src/vggt/jax/aggregator.py:307`) emits, per frame, a full token set of shape
`(1374, 2048)` = `1 camera + 4 register + 1369 patch` tokens (37×37 patch grid).
The extractor splits each token on the channel axis into a frame half and a
**global half**; `global_tokens = (1374, 1024)` is that global half
(`src/vggt/jax/feature_extractor.py:947`), cast to f32 at the boundary and stored
as f16 in replay (`token_adapters.py:69`).

Today the agent's persistent house context comes from a *different* VGGT output —
the DPT head's `world_points` — accumulated into an external voxel buffer
(`HouseContextPoseBuffer`) and pooled by a PointNet encoder. That path forces the
external-memory question (`investigate-vggt-scene-memory`: hash table vs graph).
The global tokens are an alternative source: a compact per-frame scene code that
already exists upstream of the point head, whose cross-episode persistence is
handled by VGGT's own KV cache under `ResetMode.PERSIST_SCENE`
(`feature_extractor.py:611`, saved/restored per `scene_id`).

An encoder for this path already exists — `HouseGlobalEmbeddingEncoder`
(`src/r2dreamer/encoders/mlp.py:348`, type `"vggt_house_global_embedding"`) — which
carries both a camera-token side branch and an RGB conv branch, concatenating to
`2048`. This change makes the **RGB conv branch** the second contributing branch in
place of the camera token, keeping the concat-to-`2048` shape: RGB conv `(…, 1024)`
⊕ patch-token PointNet `(…, 1024)`.

## Goals / Non-Goals

**Goals:**
- Use VGGT Aggregator **global patch tokens** as the live house-context signal for
  the world model, replacing the external point-cloud path.
- Reduce the `(1369, 1024)` patch tokens to a single `(1, 1024)` house embedding
  with a permutation-invariant PointNet-style reducer (max-pool only).
- **Fuse that house embedding with an RGB conv embedding** (hybrid), matching the
  `live-house-context` diagram: RGB `(…, 1024)` ⊕ house `(…, 1024)` → `(…, 2048)`.
- Rely on `ResetMode.PERSIST_SCENE` (heads-off) for cross-episode scene memory —
  no external accumulation buffer.
- Keep the RSSM/fusion contract unchanged: the `(…, 2048)` fused embedding is just
  another obs embedding.

**Non-Goals:**
- No change to the training loop, action head, or `R2RSSM` internals.
- Not re-deciding the point-cloud memory mechanism — that is
  `investigate-vggt-scene-memory`.
- **Not** committing to multi-token / attention-pooled context yet (deferred
  ablation — see Open Questions). The first version is a single max-pooled vector.
- Not re-deriving VGGT internals; the Aggregator and KV cache are used as-is.

## Decisions

### D1 — Global tokens as house context, not `world_points`
The global tokens are a compact scene code produced *before* the DPT head. Using
them removes the entire external point-accumulation problem (voxel dedup, capacity
caps, hash-vs-graph). Scene persistence moves into VGGT's KV cache, which is
already designed to carry state across episodes.
*Alternative (rejected for this change):* keep the point-cloud path — but that is
the existing baseline and the subject of a separate investigation.

### D2 — Keep only the 1369 patch tokens; drop camera + register tokens
The 4 register tokens are already dropped upstream (`constants.py:12-15`). The
**camera token** is dropped here: the patch tokens carry the spatial scene content;
the camera token encodes viewpoint, which is (a) not "house context" and (b)
available to the agent through other observations if needed. Dropping it makes the
reducer a clean set-over-patches with a `(…, 1024)` house branch; the encoder's
second branch is the RGB conv (D7), not the camera token.

The obs contract is therefore exactly **`image` + `global_patch_tokens`**: the
camera token is not emitted into replay, not declared in the agent obs shape, and
has no branch in the module. Today the branch is merely *shadowed* by the RGB branch
(`mlp.py:379-381`) while the adapter still emits the token (`token_adapters.py:294`)
— that intermediate state is what task 2.3 removes.
*Alternative (rejected):* keep the camera side branch, or keep emitting the token
while leaving the branch inert. Rejected: a shadowed branch is dead code that still
costs ~2 KB/step f16 of replay for a key nothing reads, and it makes the obs contract
lie about what the encoder consumes. Re-adding the token for the deferred camera A/B
is a contained change.

### D3 — Reducer = PointNet shared-MLP + single max-pool (no mean, no flatten)
Patch tokens are an unordered *set* of scene locations, so the reducer must be
permutation-invariant. Chosen: shared per-token MLP (`Dense → RMSNorm → SiLU`,
identical weights per token, no token interaction) → **max-pool over the 1369
tokens** → `Dense(1024)`. This is PointNet (Qi et al., arXiv:1612.00593 Eq.1),
already implemented as `TokenReducer` at `mlp.py:312-345`.
*Alternatives considered:*
- **Mean pool** — permutation-invariant too, but averages away salient structure;
  max keeps the strongest evidence per channel. Max-only is the deliberate choice.
- **Flatten `1369·1024` → `Dense`** — rejected: ~1.4B params and breaks
  permutation-invariance.
- **Attention pool / PMA / learned queries** — more expressive, learns *which*
  patches matter, but adds a cross-attention block and is the natural K-token
  variant. Deferred (Open Questions), not first version.

### D4 — Scene memory via `ResetMode.PERSIST_SCENE`, DPT heads OFF
`PERSIST_SCENE` saves/restores the Aggregator KV cache per `scene_id`, so the
attention stream (and thus the global tokens' scene context) resumes across
episodes of the same house. The token path does not use the DPT point head, so it
runs **heads-off** (`token_adapters.py:148`) — cheaper and self-consistent (no
unused point-head cache to bound).
*Alternative (rejected):* `ResetMode.FULL` — wipes per episode, defeating the
persistent-house-context goal.

### D5 — Reuse `HouseGlobalEmbeddingEncoder`, minus the camera branch, plus RGB
Rather than a new encoder from scratch, start from the existing
`"vggt_house_global_embedding"` encoder, remove the camera side branch, and keep its
RGB conv branch. The result is a hybrid: RGB conv `(…, 1024)` ⊕ patch
max-pool `(…, 1024)` → `concat → (…, 2048)`. Keeps the change small and grounded in
working code (both branches already exist in `encoders/`). Note the RGB branch here
is a direct `ConvEncoder(name="rgb")` (`mlp.py:380`), *not* the
`make_rgb_conv_encoder` helper that `HybridEncoder` uses at `mlp.py:498` — so the
conv knobs threaded there are not threaded here (D7).

### D6 — Fusion / RSSM unchanged
The `(…, 2048)` fused embedding (RGB conv ⊕ PointNet house) is concatenated with the
other observation embeddings and fed to `R2RSSM.observe` exactly as today; the
posterior "MLP" is the RSSM's own `obs_net` head (`rssm.py:240-249`), verified
identical to the `NM512/r2dreamer` reference. No world-model code changes.

### D7 — Hybrid: fuse an RGB conv branch with the house embedding
The `live-house-context` diagram pairs the global-token house embedding with a
per-step RGB conv embedding, fused before the RSSM. The RGB branch gives the model
local, per-step visual detail that the single max-pooled house vector discards; the
house vector stays the global scene prior. The RGB conv emits `1024` so the fused
width is a clean `2048`, matching the existing `HouseGlobalEmbeddingEncoder` /
`HybridEncoder` output width — no RSSM change.
*Implementation note:* that `1024` is **not** a projection. The branch is
`ConvEncoder(name="rgb")` at defaults (`mlp.py:380`); `embed_dim is None`, so the
`proj` Dense is skipped and the width falls out of the conv stack itself
(`depth 16 × mults[-1] 4` = 64 channels × 4 × 4 spatial = 1024, `cnn.py:68-85`).
`make_rgb_conv_encoder` takes no `embed_dim` (`cnn.py:88-103`). The two branch widths
are independent defaults that happen to agree; task 2.6a pins the RGB branch to
`(…, 1024)` so the fusion cannot silently desync.
*Alternative (rejected):* leave RGB decode-only (encoder token-only, `(1, 1024)`)
and let the decoder reconstruct it. Rejected: the diagram calls for RGB to be
*encoded* into the posterior, not only reconstructed.

## Risks / Trade-offs

- **Single-vector bottleneck** → collapsing a whole house to one 1024 max-pooled
  vector discards spatial layout and all but the per-channel max. Mitigation: the
  RGB conv branch (D7) now encodes per-step local detail directly into the
  posterior; the house vector is a global prior. If shown insufficient, the K-token
  variant (Open Questions) is the escalation — kept as a separate experiment, not
  pre-built.
- **Dropping the camera token may remove useful viewpoint signal** → Mitigation:
  camera pose is available through other observation keys; the camera-token
  contribution is a clean 1-line ablation if the loss shows up.
- **`PERSIST_SCENE` cache fidelity** → cross-episode resume depends on the KV cache
  faithfully carrying scene state; drift/eviction could weaken old context.
  Mitigation: this is measurable and overlaps with the characterization already
  scoped in `investigate-vggt-scene-memory` (FULL vs PERSIST_SCENE).
- **Token dtype** → tokens are stored f16 in replay and exposed f32; max-pool is
  robust to f16 but the reducer should be checked under the codebase bf16 default,
  not assumed stable.

## Open Questions

- **Single `(1, 1024)` vs K context tokens.** The deferred ablation: does one
  max-pooled vector carry enough house context, or should the reducer emit K
  tokens (attention-pool / PMA / learned queries) to preserve spatial structure?
  Decision: ship the single-vector max-pool first; run 1-vs-K as a separate
  experiment rather than confounding it into the first version.
- **Is the camera token worth keeping?** A/B the camera side branch (`(1,1024)` vs
  `2048`) once the base path trains.
- **Does `PERSIST_SCENE` measurably beat `FULL` for the token path?** Shares a
  measurement harness with the point-cloud characterization.
