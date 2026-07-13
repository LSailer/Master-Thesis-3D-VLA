## Why

`R2DreamerAgent.__init__` (in `src/r2dreamer/agent.py`) eagerly instantiates
the full parameter pytree for every encoder, the RSSM, the projector, and all
heads by running a **dummy forward**: it builds a zeros observation via
`_dummy_encoder_obs(cfg)`, calls `encoder_mod.init` + `encoder_mod.apply`, and
reads `embed.shape[-1]` just to discover `embed_size`. That dummy forward runs
on every agent construction — `launch/train.py`, every profiling script, every
test — even though the real first training/acting observation is available one
call later. `from_checkpoint` is worse: it pays for the entire dummy init of
all modules and then **overwrites** `params` / `slow_critic_params` with the
checkpoint, throwing the freshly-initialised weights away.

The dummy exists only to discover a shape. Shape is independent of input
content, and the real first batch is available at the first `train_step` /
`act` call. Lazy-initialising parameters on that first real batch deletes the
dummy forward from the construction path, deletes the wasted init in
`from_checkpoint`, and makes `embed_size` fall out of the real first
observation instead of a synthesised one. No numerics change: the parameter
pytree is identical, just materialised later.

## What Changes

- `R2DreamerAgent.__init__(config, rng_key)` becomes a **shell**: it builds
  the Flax module *objects* (`_make_encoder`, `_make_rssm`, projector, heads,
  optional decoder), the `self._modules` dict, the LaProp optimizer *builder*
  (`self.tx`), `ReturnEMA`, the acting state, and JITs the pure train/act
  functions — none of which need parameters. It stashes `rng_key` as
  `self._init_rng`, sets `self._initialized = False`, and leaves
  `self._train_state = None`.
- Add a single lazy-init entrypoint
  `initialize(self, seed_obs, *, params_override=None)`: splits `_init_rng`
  to init encoder/RSSM/projector/heads/decoder, runs
  `encoder_mod.apply(enc_params, seed_obs)` on the **real** `seed_obs` to set
  `self.embed_size`, builds the `params` dict (or adopts `params_override`),
  builds `opt_state = tx.init(params)`, `slow_critic_params`, `ema_state`,
  and assembles `self._train_state`. Sets `self._initialized = True`.
- `train_step` auto-initialises on the first call from `batch.obs`; `act` /
  `act_with_state` auto-initialise from the batched live observation. Both
  guard on `self._initialized`.
- `from_checkpoint` is rewritten to: build the shell, synthesise a
  **shape-only** zeros observation from `obs_shape` (the contract or arg),
  call `initialize(seed_obs, params_override=ckpt["params"])`, then assign
  `slow_critic_params` from the checkpoint. It returns a ready agent, as
  today. The per-encoder-type `_dummy_encoder_obs` helper is retained **only**
  for this checkpoint path (Flax requires running `encoder_mod.init` on some
  input to materialise the param tree structure before overwriting values);
  it is no longer called on the training/acting construction path.
- Update `scripts/debug_viz/evaluate_debug.py` (the one manual
  construct-then-assign-params clone of `from_checkpoint`) to use
  `from_checkpoint` or `initialize`.
- Update the handful of tests that read `agent.embed_size` (or otherwise touch
  `params`) before the first step to call `agent.initialize(...)` first.

## Capabilities

### New Capabilities
- `agent-construction`: the `R2DreamerAgent` construction and parameter
  initialisation contract — what `__init__` guarantees (a parameter-less
  shell), what `initialize` does (materialises the param pytree from a real
  observation, optionally overriding params from a checkpoint), and the
  first-call auto-init guarantee of `train_step` / `act`.

### Modified Capabilities

<!-- none — main specs are empty; the change introduces the capability. -->

## Impact

- `src/r2dreamer/agent.py`: `__init__` split into shell + `initialize`;
  `train_step` / `act` / `act_with_state` gain a first-call init guard;
  `from_checkpoint` rewritten. The `train_state` / `params` / `opt_state` /
  `slow_critic_params` / `ema_state` property setters must tolerate
  `self._train_state is None` until `initialize` runs (or `from_checkpoint`
  must call `initialize` before the setters do, which it does).
- `scripts/debug_viz/evaluate_debug.py`: replace manual
  construct-then-assign with `from_checkpoint` / `initialize`.
- `tests/r2dreamer/test_agent.py`, `tests/r2dreamer/test_vggt_encoder.py`,
  `tests/r2dreamer/test_trainer.py`,
  `tests/r2dreamer/world_model/test_hybrid_encoder.py`,
  `tests/r2dreamer/test_habitat_act_state_parity.py`,
  `tests/r2dreamer/test_decoder_probe_overfit_gpu.py`: insert
  `agent.initialize(...)` before any pre-step read of `embed_size` / `params`.
- No change to `encoders/factory.py`, the Flax encoder modules, the loss
  composition, checkpoint *format*, or any runtime numeric. The parameter
  pytree structure is identical to today; only the *timing* of its
  materialisation changes.
- GPU-marked tests (`test_decoder_probe_overfit_gpu.py`, any Habitat/VGGT
  end-to-end) run under `srun`/sbatch per `AGENTS.md`; CPU-safe unit tests run
  locally.