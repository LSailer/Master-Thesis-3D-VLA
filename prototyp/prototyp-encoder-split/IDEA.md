# Encoder split: move encoder ownership out of the Dreamer

Follow-up to ADR 0006 (env + buffer decoupled behind `ExperienceCollector`).
This prototype moves the remaining coupling out of `R2DreamerAgent`: encoder
construction/ownership, and the monolithic agent block.

Full design with diagrams and rationale: open `design.html` in a browser
(self-contained, works offline). Decision log is the topmost card (8a).

## Target architecture

```
env ──ObservationFrame──► ObsAdapter (frozen: VGGT, house-points, symlog)
        │                          │
        ▼ (acting)                 ▼ (collect)
   encoder_obs [1,·]          replay_obs ──► ReplayBuffer ──sample──► batch.obs [B,T,·]
        │                                                                │
        ▼                                                                ▼
   enc.apply ──f32[1,E]──► dreamer.policy            train_step: enc.apply inside WM loss
                                                     → f32[B,T,E] → RSSM → heads
```

## Decisions (binding)

1. **Three modules** — Encoder / WorldModel / Behavior. Ownership + wiring in
   `launch/train.py`; loss composition is a free jit function, not a method.
2. **Three optimizers** — WM (incl. encoder) / Actor / Critic, each the
   existing LaProp from `src/shared/optim.py` with today's hyperparameters.
   LaProp + AGC are per-leaf, so with identical hyperparameters this is
   update-identical to the current single optimizer (golden run must prove it).
3. **Params flow** — main holds the TrainState and checkpoints it;
   `train_step` is pure (params in → params out). `enc_params` is an argument
   of the WM loss fn; one `jax.value_and_grad(..., argnums=(0, 1))` over
   (enc, wm). Actor/critic losses operate on `sg(feat)` and never reach the
   encoder.
4. **One generic `CompositeEncoder`** (Flax) — branches dict (obs key →
   mechanism module: Conv/MLP/PointNet/GNN/TokenTransformer) + fusion strategy
   (`concat` | `gate` | `concat_mlp`). Combinations are `CompositeSpec` data,
   not classes. The spec is static → the branch loop unrolls at trace time.
5. **Obs contract** — the encoder never sees `ObservationFrame`. The ObsSpec
   is inferred from the first prepared frame (`collector.reset()`), which also
   serves as the `enc.init` dummy. Single startup check:
   `set(composite.branches) == set(inferred keys)`.
6. **`EncoderRecipe` registry** — `{make_adapter, composite}` per encoder
   type. Selecting the encoder type selects the matching adapter; the
   collector receives the adapter via constructor injection (ADR 0006
   collector unchanged).
7. **Decoder stays in the WorldModel**, reconstructing from RSSM feat
   (posterior), paper-conform. RGB targets travel as `decoder_targets` in the
   batch (from `replay_obs`, as `decoder_targets.py` does today); recon loss
   is part of the WM loss.
