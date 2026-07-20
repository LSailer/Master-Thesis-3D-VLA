## Why

The R2Dreamer world model currently receives its persistent house context as an
external **point cloud** (VGGT DPT head → `world_points` → `HouseContextPoseBuffer`
→ PointNet), which requires accumulating and deduplicating points across frames and
raises the open memory-mechanism question tracked in `investigate-vggt-scene-memory`.
VGGT already produces a compact per-frame scene representation *before* the DPT
head — the Aggregator's **global tokens** — and its streaming KV cache under
`ResetMode.PERSIST_SCENE` already carries scene state across episodes. Using the
global tokens directly as house context sidesteps the external point-accumulation
problem entirely: no voxel buffer, no dedup, no hash-vs-graph decision — the scene
memory lives in VGGT's own cache and the context is a single learned embedding.

## What Changes

- **New house-context path from Aggregator global tokens** instead of DPT
  `world_points`. Input is the global half of the Aggregator output,
  `(1374, 1024)` (`src/vggt/jax/feature_extractor.py:947`).
- **Token selection:** drop the 4 register tokens and the camera token; keep only
  the **1369 patch tokens** `(1369, 1024)` (`src/r2dreamer/encoders/constants.py:12-15`).
  The obs contract is exactly `image` + `global_patch_tokens`.
- **Reducer:** PointNet-style — shared per-token MLP (`Dense → RMSNorm → SiLU`,
  same weights every token, no token interaction) → **single max-pool** over the
  1369 tokens → `Dense(1024)` projection → **`(…, 1024)`**. No mean branch.
- **Reuse-with-modification:** `HouseGlobalEmbeddingEncoder`
  (`src/r2dreamer/encoders/mlp.py:348`, type `"vggt_house_global_embedding"`)
  already implements the patch-token max-pool **and** an RGB conv branch
  (`ConvEncoder(name="rgb")`, width `1024`). This change **drops the camera branch**,
  leaving the RGB conv as the second branch, so the encoder is a hybrid: RGB
  `(…, 1024)` ⊕ patch-token PointNet `(…, 1024)` → `concat → (…, 2048)`. Today the
  camera branch is only shadowed by the RGB branch, and its token is still emitted
  into replay unread — both go.
- **Scene memory via `ResetMode.PERSIST_SCENE`** (KV cache saved/restored per
  `scene_id`), with **DPT heads OFF** — consistent, since the token path never
  touches the point head (`src/r2dreamer/adapters/token_adapters.py:148`).
- **No world-model / RSSM change:** the `(…, 2048)` fused embedding is concatenated
  with the other observation embeddings and consumed by `R2RSSM` exactly as today;
  the posterior "MLP" is the RSSM's own `obs_net` head, not a new module.

## Capabilities

### New Capabilities
- `global-token-house-context`: The observation-side mechanism that turns VGGT
  Aggregator global patch tokens into a `(…, 1024)` house embedding and fuses it
  with an RGB conv embedding into a `(…, 2048)` observation embedding for the world
  model — token selection (1369 patch tokens), the PointNet max-pool reducer, the
  RGB conv branch, the `PERSIST_SCENE` (heads-off) scene-memory contract, and the
  encoder/adapter output contract. It parallels the point-cloud house-context path
  rather than replacing it.

### Modified Capabilities
<!-- None. openspec/specs/ has no established capability specs; the point-cloud
     path and HouseContextPoseBuffer are used only as the comparison baseline and
     their requirements are unchanged by this investigation. -->

## Impact

- **New / modified code (prototype-scoped):** a hybrid house-context encoder (an
  RGB-conv + patch-token variant of `HouseGlobalEmbeddingEncoder`, camera branch
  removed) and its obs adapter; prototyping under `src/prototyp/<feature>/` per
  project convention.
- **Read / exercised, not changed:** `src/vggt/jax/feature_extractor.py`
  (Aggregator global tokens, `ResetMode.PERSIST_SCENE`, KV cache),
  `src/r2dreamer/adapters/token_adapters.py` (token extraction, heads-off path),
  `src/r2dreamer/encoders/mlp.py` (`HouseGlobalEmbeddingEncoder` baseline),
  `src/r2dreamer/world_model/rssm.py` (`R2RSSM`, unchanged fusion).
- **Relationship to other changes:** an alternative to the point-cloud
  accumulation studied in `investigate-vggt-scene-memory`; if this path is adopted,
  that investigation may become moot (an explicit archive decision, not part of
  this change).
- **No training-loop or agent-behavior change** is proposed beyond adding the new
  observation encoder; scope is the house-context representation only.
