## Context

`R2DreamerAgent.__init__` (in `src/r2dreamer/agent.py`) does two things that
require an encoder input today: (1) `encoder_mod.init(k, dummy_obs)` to
materialise the encoder parameter tree, and (2) a follow-up
`encoder_mod.apply(enc_params, dummy_obs)` whose only purpose is to read
`embed.shape[-1]` into `self.embed_size`. The dummy is `_dummy_encoder_obs(cfg)`
— a per-`encoder_type` cascade of zeros arrays/dicts. `embed_size` then drives
the Projector (`out_dim=embed_size`) and the `feat0` / `embed0` zeros used to
init the RSSM and heads.

`from_checkpoint` makes this wasteful: it calls `cls(config, init_key)` (full
dummy init of every module) and then overwrites `agent.params` and
`agent.slow_critic_params` with the checkpoint. The dummy-init weights are
discarded; only the *structure* (and `opt_state`, built from the dummy params)
is kept.

The real first observation is available one call after construction — at the
first `train_step(batch, rng_key)` (replay batch, shape `(B, T, …)`) or the
first `act(encoder_obs, …)` (single live obs, shape `(1, …)`). Flax parameter
trees are shape-independent once initialised (Dense weights are `(in, out)`,
conv kernels are `(kh, kw, in, out)`), so initialising from a `(1, …)` acting
obs and later training with `(B, T, …)` produces identical parameters; the JIT
boundary already retraces for new shapes.

## Goals / Non-Goals

**Goals:**
- Delete the dummy `.apply` forward and the `_dummy_encoder_obs` call from the
  training/acting construction path; discover `embed_size` from the real first
  observation.
- Delete the wasted full init in `from_checkpoint` (params come from the
  checkpoint, not from a dummy).
- Preserve the exact parameter pytree structure and all runtime numerics —
  this is a *timing* refactor, not a behaviour change.
- Keep `from_checkpoint` returning a ready agent (no caller-side deferred
  materialisation).

**Non-Goals (separate changes, do not bundle):**
- Do not rename the private factory imports (`_make_encoder`, `_make_rssm`,
  `_dummy_encoder_obs`, `_compute_dtype_kwargs`) to public — smell #1 from the
  explore session.
- Do not collapse the three parallel `encoder_type` dispatches (class
  resolution / construction / dummy-obs) — smell #2.
- Do not make encoders self-describe an `out_dim` property (the alternative
  "option B" from the explore session) — that is a different, complementary
  change; this change does not require it.
- Do not introduce a typed encoder/RSSM interface replacing the scattered
  `module.apply(..., method=…)` calls — smell #4.
- Do not remove the redundant `_resolve_encoder_cls` re-derivation — smell #5.
- No change to the checkpoint *format*, loss math, optimizer, EMA, or acting
  logic.

## Decisions

- **Lazy init on the first real batch for `train_step` / `act` (option A2).**
  `__init__` produces a parameter-less shell; `initialize(seed_obs)` runs once
  on the first real observation. Why: the dummy exists only for shape
  discovery, and the real obs is available one call later. Alternative
  considered: "option B" — encoders declare an `out_dim` property so no
  forward is needed at all — rejected for *this* change because it does not
  remove the `encoder_mod.init` call (Flax still needs an input to build the
  param tree) and would require per-encoder `out_dim` plumbing across ~10
  modules; A2 removes both the dummy *and* the wasted `from_checkpoint` init
  in one move. B remains viable as a follow-on that would then let
  `from_checkpoint` skip its shape-only init too.
- **`from_checkpoint` keeps a shape-only zeros init (A2, not A1).** At
  checkpoint-load time there is no real observation, and Flax requires
  `encoder_mod.init` to run on *some* input of the right shape/structure to
  materialise the param tree before its values are overwritten by the
  checkpoint. So `from_checkpoint` synthesises a zeros obs via the existing
  `_dummy_encoder_obs(cfg)` and calls
  `initialize(seed_obs, params_override=ckpt["params"])`. It returns a ready
  agent. Alternative considered: "A1" — `from_checkpoint` returns a shell
  holding the checkpoint as an override and materialises on the first
  `act`/`eval_loss` — rejected because it breaks every eval/test caller that
  reads `agent.params` before stepping and gains little (the dummy is already
  gone from the hot path).
- **One entrypoint: `initialize(self, seed_obs, *, params_override=None)`.**
  Both the auto-init guards and `from_checkpoint` route through it, so there is
  a single place that splits the init RNG, builds the param pytree, and
  assembles `train_state`. `params_override` lets `from_checkpoint` adopt the
  checkpoint's weights without a separate assign path.
- **Init RNG is stashed in `__init__`, not borrowed from the first step.**
  `self._init_rng = rng_key` is set at construction; `initialize` splits it.
  This keeps parameter initialisation deterministic w.r.t. the construction
  key (matching today) and decouples it from the first step's `rng_key`, so
  the first step's RNG is used only for its own stochasticity (dropout,
  sampling), not for weight init.
- **`from_checkpoint` calls `initialize` *before* the `slow_critic_params`
  setter.** The property setters do `self._train_state._replace(...)`, which
  fails while `self._train_state is None`. The rewritten `from_checkpoint`
  order is: build shell → `initialize(seed_obs, params_override=ckpt["params"])`
  (builds `_train_state`) → assign `slow_critic_params` → set `checkpoint_step`.
- **`_modules` stays eager.** `heldout_eval.py` reads `agent._modules["reward"]`
  before any step. The `_modules` dict holds Flax module *objects*, which do
  not need parameters, so it is built in `__init__` as today and remains
  available pre-step.
- **Config-only validation stays eager in `__init__` / module construction.**
  Two errors are raised *during construction* today and tests assert that:
  `decoder=True requires an RGB-bearing encoder_type` (raised in `__init__`
  from a pure config check) and `hybrid obs_shape/split mismatch` (raised
  inside `_make_hybrid_encoder`, which the shell still calls eagerly to build
  the module object). Both depend only on config, not on parameters, so they
  stay where they are. Only *parameter-materialisation* (the `init`/`apply`
  forward, `opt_state`, `slow_critic`, `train_state`, `embed_size`) moves to
  `initialize`. This keeps `test_vggt_plus_decoder_raises` and
  `test_hybrid_split_mismatch_raises_value_error` green unchanged; only
  `test_cnn_and_hybrid_plus_decoder_build` (which reads `"decoder" in
  a.params` post-construction) needs an `initialize` call.
- **`embed_size` tests use `_dummy_encoder_obs(cfg)` to drive `initialize`.**
  The few tests that assert `agent.embed_size` right after construction call
  `agent.initialize(_dummy_encoder_obs(cfg))` first. Shape is content-
  independent, so a zeros obs yields the same `embed_size` as a real one; the
  dummy survives only in the checkpoint + embed_size-test paths, not the
  production hot path.

## Risks

- **First-call latency and hidden state mutation.** The first `train_step` /
  `act` now also runs full parameter init (slower) and flips
  `self._initialized`. Mitigation: one-time cost; documented on `initialize`
  and on the guards. No re-entrancy concern (single-threaded training).
- **`act` initialises from a `(1, …)` obs, `train_step` later from `(B, T, …)`.**
  Parameters are shape-independent → identical weights; JIT retraces for the
  new shapes, which it already does today. No correctness impact.
- **`params_override` structure must match `initialize`'s freshly-built tree.**
  Flax param trees are deterministic given module + input shape, so the tree
  built by `initialize` matches the one the checkpoint was saved from. Module
  version drift is the same risk as today (`load_checkpoint` already tolerates
  moved optimizer classes). Mitigation: the existing
  `test_agent_from_checkpoint_recovers_encoder_contract_when_shape_omitted`
  test and the round-trip save/load tests cover this.
- **Determinism tests comparing two agents (`test_agent.py` agent_a/agent_b).**
  These rely on identical init RNG producing identical params. Since
  `initialize` splits the stashed `_init_rng` identically, two agents built
  with the same `rng_key` and driven with the same first obs produce identical
  params. Mitigation: task 4.4 verifies these tests stay green.
- **Property setters while `_train_state is None`.** Any caller that assigns
  `agent.params = …` / `agent.slow_critic_params = …` before `initialize` will
  hit `None._replace`. Mitigation: `from_checkpoint` is fixed to call
  `initialize` first; `evaluate_debug.py` is migrated off the manual pattern;
  the design adds no other pre-init assignment paths.

## Verification

The safety net is **no-behavior-change**: the existing test suite exercises
`train_step`, `act`, `from_checkpoint`, save/load round-trip, and
`embed_size` across the encoder_type matrix. Greens there prove the lazy
materialisation is observationally identical to eager init.

- CPU-safe: `test_agent.py`, `test_vggt_encoder.py`, `test_trainer.py`,
  `test_hybrid_encoder.py`, `test_habitat_act_state_purity.py` — run locally.
- GPU-marked: `test_decoder_probe_overfit_gpu.py` and any Habitat/VGGT
  end-to-end — run under `srun`/sbatch.
- A leaf-for-leaf param-structure equivalence check (task 4.1) for a
  representative encoder_type set (`cnn`, `vggt`, `hybrid`,
  `vggt_house_points_pose`, `vggt_house_global_embedding`) proves the
  `initialize`-built pytree matches the pre-change eager pytree.