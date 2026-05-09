# Variant 1 Aggregator MLP Learning Log

- 2026-05-09T04:27:47Z
  - Tried: Created a clean worktree from `origin/main` on branch `feat/variant-1-aggregator-mlp` and inspected the existing VGGT and `feat/vggt-film-encoder-109` encoder patterns.
  - Chose: Keep the new variant as a separate registry encoder rather than altering the existing `vggt` WP+CP path.
  - Why: The architecture is locked to pre-head aggregator tokens, and the user constrained changes to the new encoder plus minimum plumbing. `git diff origin/main..origin/feat/vggt-film-encoder-109` showed the expected style: dataclass config fields, registry entry, RMSNorm+SiLU Flax module, and a thin launcher class.

- 2026-05-09T04:27:47Z
  - Tried: Traced `JAXVGGTFeatureExtractor.extract()` through `modules/vggt/jax/aggregator.py` and `modules/vggt/jax/heads/dpt_head.py`.
  - Chose: Export patch tokens from the last aggregator block before camera/point heads, not the existing world-points + camera-pose output. The local JAX aggregator returns per-layer tensors shaped `(B, S, P, 2048)` because it concatenates frame and global streams; the DPT head consumes those as pre-head tokens. For the locked `(B, 37, 37, 1024)` variant, use the final global half (`[..., 1024:]`) after removing the 5 special tokens.
  - Why: The user explicitly rejected WP+CP and specified pre-head aggregator features. The final global stream is the contextualized aggregator representation immediately before downstream heads, while the first half is the frame-local stream.

- 2026-05-09T04:32:49Z
  - Tried: Ran `uv run pytest modules/r2dreamer/tests/test_vggt_encoder.py modules/r2dreamer/launch/tests/test_registries.py -q` in the clean worktree.
  - Chose: Treat the first attempt as an environment fork and switch to a Python version compatible with Open3D before re-running tests.
  - Why: `uv` selected CPython 3.13 for the new worktree, but `open3d==0.19.0` publishes wheels only for cp310/cp311/cp312. The failure occurred before project tests executed.

- 2026-05-09T04:40:17Z
  - Tried: Generated curriculum files in the clean worktree after registry tests failed on missing `data/curriculum/*.json`.
  - Chose: Reuse the original checkout's local `data/` tree via worktree-local symlink for tests and smoke instead of committing generated datasets/configs.
  - Why: Curriculum generation depends on the untracked HM3D/ObjectNav dataset (`data/datasets/objectnav/hm3d/objectnav_hm3d_v2/train/train.json.gz`), which exists in the original checkout but not in a fresh git worktree. Committing or copying dataset artifacts would violate the minimal-change constraint.

- 2026-05-09T04:45:00Z
  - Tried: Ran the Variant 1 Habitat+VGGT smoke from the clean worktree.
  - Chose: Add a worktree-local `external/` symlink to the original checkout's external dependencies before re-running smoke.
  - Why: The failure was not in Variant 1 code; the JAX VGGT weight loader imports `streamvggt` from `external/InfiniteVGGT/src`, and clean git worktrees do not copy the untracked/ignored `external` dependency tree. The original checkout already has it.

- 2026-05-09T04:50:00Z
  - Tried: Re-ran smoke after restoring the external StreamVGGT dependency path.
  - Chose: Add a worktree-local `data/scene_datasets` symlink to the original checkout before the next smoke run.
  - Why: Habitat initialized and reached simulator construction, then failed because the fresh worktree did not have the HM3D scene assets under `data/scene_datasets`; this is another local data dependency absent from clean git worktrees, not a Variant 1 encoder failure.

- 2026-05-09T04:55:00Z
  - Tried: Let the smoke allocate replay after Habitat and VGGT initialized.
  - Chose: Store aggregator replay observations as float16 and cap the Variant 1 replay window at 5,000 transitions.
  - Why: The locked input shape `(37, 37, 1024)` makes a 1,000,000-step float32 replay allocate ~5.10 TiB. This is not viable on H100 nodes. float16 plus a 5k replay window keeps the Variant 1 smoke/SLURM job feasible while preserving Dreamer inputs as float32 after sampling.

- 2026-05-09T05:08:00Z
  - Tried: Let the 10k smoke continue into the first JAX train-step compile with the default Dreamer batch `(batch_size=16, seq_len=64)`.
  - Chose: Use Variant 1-specific training defaults `batch_size=4`, `seq_len=32`, and `train_ratio=128` for this memory-heavy encoder.
  - Why: The run reached episode metrics at steps 499 and 999, proving Habitat+VGGT acting and W&B logging, but the first train-step compile consumed ~75 GiB H100 VRAM. The aggregator MLP has a very large 87,616-wide flattened input, so reduced sequence/batch settings keep the ablation feasible on a single H100 while maintaining the locked encoder architecture.
