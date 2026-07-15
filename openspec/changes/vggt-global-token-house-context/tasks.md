

## 2. Reducer + RGB conv hybrid fusion

- [x] 2.1 Patch-only reducer: shared per-token MLP (`Dense → RMSNorm → SiLU`) over
      the 1369 patch tokens — `TokenReducer`, `mlp.py:312-340`
- [x] 2.2 Single max-pool over the token axis (no mean branch), then `Dense(1024)`
      projection → `(…, 1024)` house branch — `mlp.py:342,345`
- [ ] 2.3 Remove the camera side branch end-to-end; the PointNet house branch alone
      is `(…, 1024)` and the obs contract becomes exactly
      `image` + `global_patch_tokens`. Today the branch is only *shadowed* by the RGB
      branch (`mlp.py:379-381`), not removed, and the token is still emitted unread
      (~2 KB/step f16)

- [ ] 2.3d Replace `test_image_shadows_the_camera_token_when_both_are_present`
      (`test_house_global_embedding_encoder.py:72-85`) — it pins the shadowing
      behavior this task deletes — and drop
      `test_camera_obs_without_image_uses_the_camera_branch` (`:93`)
- [ ] 2.3e Re-scope 6.2: with the token gone from replay, the camera A/B is no longer
      a module-side flag — note the adapter+contract work it now implies
- [x] 2.4 Permutation invariance — `test_max_pool_is_permutation_invariant`
      (`tests/r2dreamer/test_house_global_embedding_encoder.py:203`)
- [ ] 2.5 Leading-dim preservation: `test_preserves_replay_leading_dims` (`:165`)
      covers `(B, T)` but asserts the **fused** `(2, 4, 2048)`, not the house branch
      alone. Add a house-branch-only `(B, T, 1369, 1024)` → `(B, T, 1024)` assertion
- [x] 2.6 RGB conv branch exists — but as `ConvEncoder(name="rgb")` (`mlp.py:380`),
      **not** `make_rgb_conv_encoder(embed_dim=1024)`: that helper takes no
      `embed_dim` (`cnn.py:88-103`). The 1024 width is emergent (`depth 16 ×
      mults[-1] 4` = 64 ch × 4 × 4 spatial; `proj` skipped since `embed_dim is
      None`). Config knobs `encoder_depth`/`encoder_kernel`/`encoder_mults` are
      inert on this path
- [ ] 2.6a Pin the RGB branch to `(…, 1024)` in a test at the real `3×64×64` input,
      so the `2048` fusion cannot silently desync if `depth`/`mults`/input size change
- [x] 2.7 Concatenate `[rgb_embed | house_embed]` → `(…, 2048)` fused output (the
      `+` in the diagram) — `mlp.py:381`
- [x] 2.8 Fused width `2048` asserted as `2 * BRANCH_DIM` —
      `test_house_global_embedding_encoder.py:68,80,93,181`
- [ ] 2.9 **BLOCKER — the encoder is non-constructible through the factory.**
      `module_kwargs_from_config` (`house_global_embedding.py:117-140`) emits
      `embed_dim`/`token_dim`/`num_patch_tokens`/`reducer_hidden`/`reducer_layers`/
      `camera_hidden`/`camera_layers`; the module declares only 
      `mlp_layers`/`hidden_dim` (`mlp.py:362-363`), and neither
      `_contract_encoder_kwargs` nor `encoder_module_kwargs_from_config` filters.
      `TypeError: unexpected keyword argument 'embed_dim'`;
      `test_agent.py::test_train_step_accepts_split_global_tokens_pointnet_reducer`
      fails on main today. Blocks 5.1–5.3. May belong to
      `fold-vggt-specs-onto-encoders` ("Cause A drift") rather than here
- [ ] 2.10 Decide which side owns branch width: renaming the kwargs is not enough —
      the module never threads `output_dim`/`token_dim`/`num_patch_tokens` into
      `TokenReducer` (hardcoded to the `1024` default, `mlp.py:328`), while
      `test_agent.py:479` expects `2 * vggt_embed_dim`
- [ ] 2.11 Add one production-dim test (1369×1024 tokens, `3×64×64` RGB) through the
      factory; today's unit tests bypass `_make_encoder` and construct the module
      directly (`test_house_global_embedding_encoder.py:33-36`), which is exactly why
      2.9 slipped through

## 3. Adapter: token selection + scene memory

- [ ] 3.1 Extract Aggregator global tokens `(…, 1374, 1024)` and select the 1369 patch tokens (drop camera + 4 register tokens), ref `feature_extractor.py:947`, `constants.py:12-15`
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
