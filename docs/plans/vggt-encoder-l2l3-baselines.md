# Plan: VGGT 3D Encoder + L2/L3 Curriculum Baselines

> Source PRD: #66

## Architectural decisions

- **Observation space (CNN baselines)**: `(3, 64, 64)` RGB uint8 — unchanged
- **Observation space (VGGT)**: `(4116,)` float32 — flattened world_points `(37×37×3)` + camera_pose `(9)`
- **Action space**: 4 discrete (STOP, FORWARD, LEFT, RIGHT) — unchanged
- **Encoder (VGGT)**: Single linear layer 4116 → 1024 (replaces R2Encoder CNN)
- **VGGT model**: InfiniteVGGT (StreamVGGT) in PyTorch, frozen, streaming mode with KV-cache
- **Framework boundary**: VGGT (PyTorch) → `.cpu().numpy()` → JAX arrays
- **Replay buffer (VGGT)**: Stores `(cap, 4116)` float32 features, not RGB images
- **KV-cache lifecycle**: Reset at episode boundaries (`done` flag)
- **Barlow Twins**: Compares projected RSSM feature vs VGGT embedding — mechanism unchanged
- **Render resolution (VGGT)**: 518×518 (VGGT native input size)
- **Training steps**: 2.4M for all experiments
- **Step penalty**: -0.01 for all experiments

---

## Phase 1: Enhanced SPL Logging + L1 Re-run

### What to build

Add per-episode path efficiency metrics to the training loop and re-run L1 to diagnose why SPL declines despite rising SR. End-to-end: modify logging → create SLURM script → launch → verify in WandB.

**Code changes:**
- `modules/r2dreamer/scripts/run_jax_habitat.py`: log `path_length`, `shortest_path`, `path_ratio` (actual/shortest) per episode from Habitat episode info
- Add `episode_reset` WandB event marker at episode boundaries
- Create `modules/r2dreamer/scripts/slurm/train_curriculum_l1_rerun.sbatch`

**Launch:** 2.4M steps, gpu_h100, 24h, L1 curriculum

### Acceptance criteria

- [ ] Per-episode `path_length`, `shortest_path`, `path_ratio` logged to WandB + CSV
- [ ] `episode_reset` markers visible in WandB
- [ ] L1 re-run completes 2.4M steps
- [ ] SPL trend over training is analyzable (rolling window in WandB)

---

## Phase 2: L2/L3 CNN Baselines

### What to build

Launch L2 and L3 with the existing R2-Dreamer CNN pipeline. No code changes — only new SLURM scripts with different curriculum paths. Can run in parallel with Phase 1.

**Code changes:**
- Create `modules/r2dreamer/scripts/slurm/train_curriculum_l2.sbatch` — points to `data/curriculum/level2_1house_6goals.json`
- Create `modules/r2dreamer/scripts/slurm/train_curriculum_l3.sbatch` — points to `data/curriculum/level3_10houses_1goal.json`

**Launch:** 2.4M steps each, gpu_h100, 24h each

### Acceptance criteria

- [ ] L2 baseline completes 2.4M steps, metrics in WandB + CSV
- [ ] L3 baseline completes 2.4M steps, metrics in WandB + CSV
- [ ] WandB tags distinguish L1/L2/L3 runs

---

## Phase 3: VGGT Feature Extraction Smoke Test

### What to build

Implement the `VGGTFeatureExtractor` PyTorch wrapper around InfiniteVGGT. Verify it produces correct output shapes on real Habitat frames in streaming mode.

**Code changes:**
- Create `modules/vggt/feature_extractor.py`: `VGGTFeatureExtractor` class
  - `__init__`: load frozen InfiniteVGGT via `variants.load_variant("infinite_vggt")`
  - `reset()`: clear KV-cache and frame counter
  - `extract(rgb)`: single-frame streaming inference → `{"world_points": (37,37,3), "camera_pose": (9,)}`
- Create `modules/vggt/tests/test_feature_extractor.py`: smoke test on real Habitat frames
  - Render one episode (10 frames) at 518×518
  - Verify output shapes, dtypes, value ranges
  - Verify KV-cache reset produces consistent first-frame output

**Run:** pytest on dev_gpu_h100

### Acceptance criteria

- [ ] `VGGTFeatureExtractor` loads InfiniteVGGT without errors
- [ ] `extract()` returns `world_points (37,37,3)` float32 and `camera_pose (9,)` float32
- [ ] `reset()` clears KV-cache; re-extracting the same frame after reset produces identical output
- [ ] Streaming mode: extracting frame N uses context from frames 0..N-1

---

## Phase 4: VGGT Encoder + Replay + Training Script

### What to build

Wire VGGT features into the full R2-Dreamer training pipeline. New JAX encoder, modified replay buffer, new training script. Verify end-to-end with a short smoke test.

**Code changes:**
- `modules/r2dreamer/networks.py`: add `VGGTEncoder(nn.Module)` — single Dense layer, `(B, 4116) → (B, 1024)`
- `modules/dreamerv3/replay_buffer.py`: add `VGGTReplayBuffer` class — stores `(cap, 4116)` float32 obs, no `/255.0` normalization
- `modules/r2dreamer/agent.py`: support swapping `R2Encoder` for `VGGTEncoder` via config flag (e.g. `encoder_type: str = "cnn"` vs `"vggt"`)
- `modules/r2dreamer/config.py`: add `encoder_type`, `vggt_feature_dim: int = 4116`, `render_resolution: int = 64`
- Create `modules/r2dreamer/scripts/run_jax_habitat_vggt.py`: training script with VGGTFeatureExtractor in the acting loop
  - Acting: `env.step() → extractor.extract(rgb) → flatten → replay.add() → agent.act()`
  - Episode boundary: `extractor.reset()` + log `episode_reset` to WandB
  - Training: unchanged (sample from replay, train world model on stored features)
- Create `modules/r2dreamer/tests/test_vggt_encoder.py`: shape tests for VGGTEncoder + VGGTReplayBuffer

**Run:** 5K-step smoke test on dev_gpu_h100 (verify losses decrease, no crashes)

### Acceptance criteria

- [ ] `VGGTEncoder` produces `(B, 1024)` from `(B, 4116)` input
- [ ] `VGGTReplayBuffer` stores/samples float32 feature vectors correctly
- [ ] Training script runs 5K steps without shape mismatches or crashes
- [ ] Barlow Twins loss computed on VGGT embeddings (not RGB CNN)
- [ ] Losses decrease during smoke test (world model is learning)
- [ ] `episode_reset` logged to WandB at each episode boundary

---

## Phase 5: L1 VGGT Full Training

### What to build

Launch the full 2.4M-step L1 VGGT training run. No code changes — just the SLURM script and launch.

**Code changes:**
- Create `modules/r2dreamer/scripts/slurm/train_curriculum_l1_vggt.sbatch`
  - 48h wall time, gpu_h100, 1 GPU
  - Uses `run_jax_habitat_vggt.py` with L1 curriculum
  - WandB tags: `curriculum,level1,1house,chair-only,vggt,3d-encoder`

**Launch:** 2.4M steps, gpu_h100, 48h

### Acceptance criteria

- [ ] L1 VGGT run completes 2.4M steps without crashes
- [ ] Checkpoints saved every 100K steps
- [ ] Metrics logged to WandB + CSV (SR, SPL, losses, episode_reset markers)
- [ ] VGGT features stored in replay buffer throughout training

---

## Phase 6: Reporter Comparison

### What to build

Run `/reporter` on all completed experiments. Generate cross-experiment comparison slides and wiki pages.

**Analysis:**
- **Cross-level CNN comparison** (L1 vs L2 vs L3): SR, SPL, world model losses — where does the WM struggle?
- **3D vs 2D comparison** (L1 CNN vs L1 VGGT): SR, SPL, learning curves, world model losses
- **SPL analysis** (L1 re-run): path_ratio trend over training, SPL vs SR correlation, per-episode efficiency

**Outputs:**
- Wiki experiment pages for L1-rerun, L2, L3, L1-VGGT
- HTML slides comparing all experiments
- Plot figures in `output/figures/`

### Acceptance criteria

- [ ] Wiki pages created for all new experiments
- [ ] HTML slides with cross-level comparison (L1 vs L2 vs L3)
- [ ] HTML slides with encoder comparison (L1 CNN vs L1 VGGT)
- [ ] SPL analysis plots: path_ratio trend, SPL vs SR scatter
- [ ] Wiki index and log updated
