## 1. Prototype scaffold

- [ ] 1.1 Create `src/prototyp/global_token_house_context/` per project convention (read `src/prototyp/CLAUDE.md` first)
- [ ] 1.2 Note the reuse baseline in the prototype README: `HouseGlobalEmbeddingEncoder` (`encoders/mlp.py:294`), type `"vggt_house_global_embedding"`, and what is being changed (drop camera branch)

## 2. Reducer + RGB conv hybrid fusion

- [ ] 2.1 Implement a patch-only reducer: shared per-token MLP (`Dense → RMSNorm → SiLU`) over the 1369 patch tokens
- [ ] 2.2 Apply a single max-pool over the token axis (no mean branch), then `Dense(1024)` projection → `(…, 1024)` house branch
- [ ] 2.3 Remove the camera side branch; the PointNet house branch alone is `(…, 1024)`
- [ ] 2.4 Unit-check permutation invariance: shuffling the 1369 tokens leaves the `(…, 1024)` house branch unchanged
- [ ] 2.5 Unit-check leading-dim preservation for `(B, T, 1369, 1024)` → `(B, T, 1024)`
- [ ] 2.6 Add the RGB conv branch: `make_rgb_conv_encoder(embed_dim=1024)` over `hybrid_image` `(…, 3, 64, 64)` → `(…, 1024)`
- [ ] 2.7 Concatenate `[rgb_embed | house_embed]` → `(…, 2048)` fused output (the `+` in the diagram)
- [ ] 2.8 Assert the fused encoder output width is `2048` (RGB `1024` ⊕ house `1024`)

## 3. Adapter: token selection + scene memory

- [ ] 3.1 Extract Aggregator global tokens `(…, 1374, 1024)` and select the 1369 patch tokens (drop camera + 4 register tokens), ref `feature_extractor.py:944`, `constants.py:12-15`
- [ ] 3.2 Configure `ResetMode.PERSIST_SCENE` with DPT heads off, ref `token_adapters.py:148`
- [ ] 3.3 Verify no `world_points` / point-buffer dependency on this path (no `HouseContextPoseBuffer` consulted)
- [ ] 3.4 Confirm the adapter emits `hybrid_image` (RGB `3×64×64`) alongside `global_patch_tokens`, and drops `camera_token_global`

## 4. Wiring

- [ ] 4.1 Register the new encoder type (or a `camera_branch=False` flag on the existing one) in `encoders/factory.py`
- [ ] 4.2 Wire the obs keys (`hybrid_image` + `global_patch_tokens`) so the fused `(…, 2048)` embedding is consumed by `R2RSSM` (no RSSM change)
- [ ] 4.3 Confirm the RSSM contract is untouched: `deter=2048`, `stoch=32×16`, `feat=2560`

## 5. Verification

- [ ] 5.1 Shape smoke test end-to-end: frame → (global tokens → reducer `(…, 1024)`) + (RGB → conv `(…, 1024)`) → concat `(…, 2048)` → RSSM `observe` runs
- [ ] 5.2 Confirm PERSIST_SCENE restores the KV cache across two episodes of the same `scene_id`
- [ ] 5.3 Short training smoke run (single L1 scene) to confirm the path trains without shape/dtype errors under the codebase bf16 default
- [ ] 5.4 Log the token-replay footprint (≈ 1369×1024×2 bytes/step f16) and confirm it fits the token-path batch/seq overrides

## 6. Deferred (out of scope — separate experiments, do NOT implement here)

- [ ] 6.1 Ablation stub only: note where a K-token attention-pool (PMA/learned queries) reducer would plug in, for the single-vector-vs-K experiment
- [ ] 6.2 Ablation stub only: note the camera-token A/B (add the camera token back as a third branch vs the RGB + patch hybrid)
