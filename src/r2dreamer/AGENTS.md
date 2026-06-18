# AGENTS.md — `src/r2dreamer/`

Module contract for the R2Dreamer agent. This file scopes the repo-root
[`AGENTS.md`](../../AGENTS.md) (worktree setup, GPU via `srun`, Linear/PR rules) to
this package. Read it before editing anything under `src/r2dreamer/`.

## Purpose

R2Dreamer is a DreamerV3-style, **JAX/Flax** world-model agent for embodied
ObjectNav in Habitat (and Crafter). It learns a latent RSSM dynamics model, trains
an actor–critic by imagination in latent space, and co-trains a representation
objective (Barlow Twins + replay-value). The whole agent is one params pytree
optimised by a single `jax.grad` under the LaProp + AGC optimiser.

The *library* lives here; *runnable drivers* (collection, offline training,
curriculum sbatch) live in [`scripts/r2dreamer/`](../../scripts/r2dreamer/AGENTS.md);
*tests* in [`tests/r2dreamer/`](../../tests/r2dreamer/AGENTS.md).

## Layout

```
src/r2dreamer/
├── agent.py ............ R2DreamerAgent — composition root: owns params, optimiser,
│                         slow-critic EMA, JIT'd train_step()/act(); loads checkpoints
├── config.py ........... R2DreamerConfig dataclass — every hyperparameter + size presets
├── trainer.py .......... Trainer + TrainerConfig — env loop, prefill, logging,
│                         convert_batch(), save_checkpoint()/load_checkpoint()
├── heldout_eval.py ..... offline WM probes: reward MSE, k-step rollout error, recon NLL
├── manifest.py ......... MANIFEST.json writer (git SHA, config, W&B id, SLURM id)
│
├── world_model/        latent dynamics + observation/reward heads
│   ├── rssm.py ......... RMSNorm, BlockLinear, Deter (block-GRU), R2RSSM (observe/img_step/get_feat)
│   ├── encoders.py ..... ConvEncoder, VGGTEncoder, VGGTAggregatorMLPEncoder, WPConvEncoder,
│   │                     HybridEncoder, ConvDecoder + HYBRID_RGB_DIM / HYBRID_VGGT_DIM
│   ├── heads.py ........ R2MLP, R2TwoHotDist (symexp two-hot reward/critic)
│   └── loss.py ......... world_model_loss(), kl_loss() (free-nats, asymmetric)
│
├── behavior/           actor–critic over imagined rollouts
│   ├── imagination.py .. _imagine() (detached rollout), _lambda_return() (GAE-λ)
│   ├── loss.py ......... behavior_loss() — policy + value losses
│   └── return_ema.py ... ReturnEMA — 5th/95th return percentiles for advantage scaling
│
├── representation/     self-supervised representation learning
│   ├── barlow.py ....... Projector, barlow_loss() (cross-correlation)
│   ├── repvalue.py ..... repval_loss() (replay-value bootstrap from slow critic)
│   └── loss.py ......... representation_loss() (composition)
│
├── adapters/           env-obs → buffer/agent bridge
│   ├── obs_adapter.py .. ObsAdapter base (RGB passthrough)
│   ├── vggt_adapter.py . VGGTObsAdapter + VGGT_FEATURE_DIM=4116, delegates readouts
│   └── hybrid_adapter.py HybridObsAdapter + HYBRID_FEATURE_DIM=16404
│
├── observation_preparation/ encoder-input contracts and env→replay/agent prep
│   ├── cnn.py .......... CNNObservationPreparation
│   ├── vggt.py ......... VGGT-family contract/dimension helpers
│   └── vggt_readouts.py  VGGT head/token readout adapters and pooling/flattening
│
├── encoders/__init__.py launcher-side Encoder/EncoderSpec specs (CNN, VGGT*, Hybrid)
│
└── launch/             entrypoints + wiring
    ├── train.py ........ train() — resolves env/encoder/curriculum, builds Trainer, runs
    ├── evaluate.py ..... evaluate() — load checkpoint, run episodes, log to W&B
    ├── parser.py ....... shared argparse for the launcher shims
    ├── registries.py ... encoder_registry {cnn, vggt, vggt_aggregator_mlp,
    │                     vggt_wp_dense_cnn, vggt_wp_cp_64, hybrid}, env_registry {habitat, crafter}
    ├── curricula.py .... CURRICULA {L1..L4} → data/curriculum/*.json
    ├── habitat_setup.py  make_habitat_env() (HabitatObjectNavEnv factory)
    └── parity/ ......... train_parity.py, batch_utils.py, benchmark.py — JAX↔PyTorch parity
```

## Entry points

```python
# Train (driven by the scripts/r2dreamer/run.py dispatcher via _run_configs.launch_run)
from src.r2dreamer.launch.train import train
train(env="habitat", encoder="cnn", curriculum="L1", output_dir=..., wandb_name=...)

# Evaluate a checkpoint
from src.r2dreamer.launch.evaluate import evaluate
evaluate(checkpoint_path=".../step_xxxxxx.pkl", env="habitat", encoder="cnn")

# Direct agent use
from src.r2dreamer.agent import R2DreamerAgent
from src.r2dreamer.config import R2DreamerConfig
agent = R2DreamerAgent(R2DreamerConfig(...), rng_key)
metrics = agent.train_step(batch, rng_key)   # JIT'd
action  = agent.act(obs_dict, rng_key)
```

`encoder` and `env` strings resolve through `launch/registries.py`; `curriculum`
through `launch/curricula.py`. To add an encoder: implement an `Encoder`/`EncoderSpec`
in `encoders/__init__.py`, register it in `registries.py`, then add a `RUN_CONFIGS`
entry in `scripts/r2dreamer/_run_configs.py` (launched via `run.py <run-id>`) and a
`test_presets.py` entry.

## Data flow

```
env obs ──ObsAdapter──> buffer(uint8 RGB | float32 VGGT | hybrid image+WP/CP fields)
        │
train_step(batch):
  encoder(obs) -> embed
  R2RSSM.observe(embed, actions, is_first) -> post stoch/deter   (posterior)
  R2RSSM prior(actions)                    -> prior logits        (dynamics)
  feat = get_feat(stoch, deter)
  ├─ world_model_loss : KL(dyn)+KL(rep) + reward + continue [+ decoder]
  ├─ behavior_loss    : imagine H steps (detached) -> λ-returns -> policy+value
  └─ representation_loss: barlow + repval
  total = Σ cfg.scale_* · loss  ──jax.grad──> LaProp+AGC update
  slow_critic ← EMA(critic)   (updated outside jax.grad)
```

## Conventions

- **Config-first.** `R2DreamerConfig` (a `@dataclass`) is the single source of truth;
  CLI flags in `launch/parser.py` override it. No YAML.
- **CHW, JAX layout.** `obs_shape = (C, H, W)`. RGB is `/255` then centred to `[-0.5, 0.5]`
  inside `ConvEncoder`; VGGT/hybrid vectors are **not** `/255` (metric XYZ / learned tokens).
- **Params are a plain dict pytree** (`encoder`, `rssm`, `projector`, `reward`, `cont`,
  `actor`, `critic`, optional `decoder`); modules are stateless Flax `nn.Module`s.
- **JIT + PRNG.** `train_step`/`act` are `jax.jit`-compiled; always `jax.random.split` keys.
  `jax.lax.stop_gradient` gates imagination, the Barlow embed, and the prior in KL.
- **Checkpoints are pickle** of `params + opt_state + slow_critic_params + ema_state + step`
  plus a JSON-serializable `encoder_input_contract` snapshot when available;
  `_CheckpointUnpickler` tolerates moved/renamed optimiser classes.

## Dependencies

- Internal: `src.shared.optim` (`laprop`, `agc`), `src.shared.video_utils`,
  `src.shared.configs.DreamerConfig`, `src.buffer.replay_buffer`,
  `src.vggt.jax.feature_extractor.JAXVGGTFeatureExtractor`,
  `src.environments.{habitat,crafter}`.
- External: `jax`, `flax.linen`, `optax`, `numpy`, `wandb` (optional), `habitat_sim` (ObjectNav only).

## Running & testing

GPU is required whenever `jax`/`habitat_sim` actually executes — wrap with `srun`
(see root `AGENTS.md`). CPU-only unit tests need no GPU:

```bash
uv run pytest tests/r2dreamer/ -m "not gpu" -q          # CPU suite
srun --partition=dev_gpu_h100 --gres=gpu:1 --time=00:30:00 \
  uv run pytest tests/r2dreamer/ -m gpu                 # GPU suite
```

## Gotchas / read-this-first

- **`is_first` must be truthful.** The RSSM zeroes `stoch`/`deter` on episode boundaries;
  a mislabelled buffer silently diverges. `act()` resets internal state on `is_first`.
- **Encoder ↔ adapter ↔ obs_shape must agree.** Hybrid replay stores explicit
  `image` and `wp_cp` fields, but the agent packs them into
  `(HYBRID_RGB_DIM + HYBRID_VGGT_DIM,) = (16404,)` before the encoder; the CNN
  encoder rejects `vggt_mlp_layers != 1`. Mismatches surface as shape checks in
  `_make_encoder()` or `obs_batch`.
- **KL is asymmetric (DreamerV3).** Dynamics KL detaches the posterior (trains the prior);
  representation KL detaches the prior (trains the encoder). Both clipped to `cfg.kl_free`.
- **Imagination is fully detached** — reward/cont/RSSM are read under `stop_gradient`; the
  critic bootstraps from the **slow** EMA critic, not the live one.
- **Actions:** buffer stores int32; `convert_batch()` one-hots to float `(B,T,A)` for the agent.
- **Parity** against the PyTorch reference (`external/r2dreamer/`) lives in `launch/parity/`;
  use it to debug numerical drift. Tolerances: ~1e-4 per-op, ~2e-3 composed.
