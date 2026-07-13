## 1. Split `__init__` into a parameter-less shell

- [ ] 1.1 In `src/r2dreamer/agent.py`, trim `__init__` to build only what needs
      no parameters: `self.cfg`, `self.checkpoint_step`, `self.twohot`, the
      Flax module *objects* via `_make_encoder` / `_make_rssm` / `Projector` /
      `R2MLP` heads / optional `ConvDecoder`, the `self._modules` dict, the
      LaProp optimizer builder `self.tx`, `self.return_ema`, the acting state
      `self._act_state`, and the JIT wrappers for `_train_step_pure` /
      `act_with_state_pure`
- [ ] 1.2 Stash `self._init_rng = rng_key`, set `self._initialized = False`,
      `self._train_state = None`, and `self.embed_size = None` (populated by
      `initialize`)
- [ ] 1.3 Confirm the `train_state` / `params` / `opt_state` /
      `slow_critic_params` / `ema_state` property getters return a clear
      `RuntimeError` ("agent not initialised; call initialize() or train_step()/
      act() first") rather than `None`-attribute errors when
      `self._train_state is None`; the setters must tolerate `_train_state is
      None` only via the `from_checkpoint` path (which calls `initialize`
      first — see 3.1)

## 2. Add `initialize(self, seed_obs, *, params_override=None)`

- [ ] 2.1 Implement `initialize`: split `self._init_rng` into the per-module
      init keys (encoder, rssm, projector, reward, cont, actor, critic, and
      decoder when `cfg.decoder`), init each module on `seed_obs` / the
      appropriate zeros seed (`feat0`, `stoch0`, `deter0`, `action0`,
      `embed0`)
- [ ] 2.2 Run `embed = encoder_mod.apply(enc_params, seed_obs)` on the real
      `seed_obs` and set `self.embed_size = embed.shape[-1]`; keep the
      `embed.shape[:2]` preservation check only where `seed_obs` carries
      replay leading dims (the `train_step` path), not for the `(1, …)`
      acting path
- [ ] 2.3 Assemble the `params` dict (encoder/rssm/projector/reward/cont/
      actor/critic, plus decoder when configured); when `params_override` is
      given, replace `params` with it (used by `from_checkpoint`)
- [ ] 2.4 Build `opt_state = self.tx.init(params)`, `slow_critic_params =
      jax.tree.map(jnp.copy, params["critic"])`, `ema_state =
      self.return_ema.init_state()`, and set
      `self.train_state = R2DTrainState(params, opt_state, slow_critic_params,
      ema_state)`; set `self._initialized = True`
- [ ] 2.5 Guard `initialize` against double-init: raise if
      `self._initialized` is already True (a silent second init would
      regenerate params and discard training state)

## 3. Auto-init guards and `from_checkpoint` rewrite

- [ ] 3.1 Rewrite `from_checkpoint`: resolve the contract / config / obs_shape
      as today, build the shell via `cls(config, init_key)`, build a
      shape-only `seed_obs = _dummy_encoder_obs(config)`, call
      `agent.initialize(seed_obs, params_override=jax.tree.map(jnp.asarray,
      ckpt["params"]))`, then assign
      `agent.slow_critic_params = jax.tree.map(jnp.asarray,
      ckpt["slow_critic_params"])` and `agent.checkpoint_step`; return the
      ready agent
- [ ] 3.2 Add `if not self._initialized: self.initialize(batch.obs)` at the
      top of `train_step` (before the JIT call); confirm `batch.obs` is the
      encoder's real input layout
- [ ] 3.3 Add the same guard to `act` and `act_with_state`, initialising from
      the *batched* live observation (`batch_live_observation(encoder_obs)`);
      confirm the `(1, …)`-shaped init produces the same params as a later
      `(B, T, …)` train path
- [ ] 3.4 Migrate `scripts/debug_viz/evaluate_debug.py` (lines ~152-154) off
      the manual construct-then-assign-params pattern onto `from_checkpoint`
      (or `initialize` with a params override)

## 4. Tests and verification

- [ ] 4.1 Add a leaf-for-leaf param-structure equivalence test for
      `encoder_type` in `{cnn, vggt, hybrid, vggt_house_points_pose,
      vggt_house_global_embedding}`: the params pytree built by `initialize`
      matches the pre-change eager pytree (shapes and dtypes; values match for
      the same init key). Capture the expected structure from the current
      `__init__` *before* the refactor lands
- [ ] 4.2 Update `tests/r2dreamer/test_agent.py`,
      `tests/r2dreamer/test_vggt_encoder.py`, and any other test that reads
      `agent.embed_size` / `agent.params` before the first step to call
      `agent.initialize(_dummy_encoder_obs(cfg))` first
- [ ] 4.3 In `tests/r2dreamer/world_model/test_hybrid_encoder.py`
      `TestDecoderGuard`: `test_vggt_plus_decoder_raises` and
      `test_hybrid_split_mismatch_raises_value_error` stay green unchanged
      (config-only validation remains eager — see design.md); only
      `test_cnn_and_hybrid_plus_decoder_build` needs
      `a.initialize(_dummy_encoder_obs(a.cfg))` (and likewise for `b`, `c`)
      before the `"decoder" in *.params` asserts
- [ ] 4.4 Confirmed pre-step reads needing `agent.initialize(_dummy_encoder_obs(cfg))`
      before the read (the `act`/`train_step` auto-init guard is too late for
      these): `tests/r2dreamer/test_agent.py:479` (reads `agent.embed_size`
      at 481 and `agent.params["encoder"]` at 496 before `train_step` at 497);
      `tests/r2dreamer/test_agent.py:548-549` (reads `agent_a.params` at 550
      before `train_step` at 552 — add `initialize` to **both** agent_a and
      agent_b with the same seed obs; this validates the stashed-init-RNG
      determinism decision); `tests/r2dreamer/test_vggt_encoder.py:477`
      (reads `agent.embed_size` and `agent.params` at 479-480);
      `tests/r2dreamer/test_vggt_encoder.py:535` (reads `agent.embed_size` at
      536 before `act` at 543). The act-only tests
      (`test_agent_act_vggt` 482-499, `test_agent_act_vggt_aggregator_mlp`
      501-518, and the `act` at 543) need no change — the `act` auto-init guard
      materialises params from the batched live observation. Cross-check
      against the grep output for any other pre-step `params` /
      `opt_state` / `slow_critic_params` / `ema_state` reads not listed here
- [ ] 4.5 Run the CPU-safe suite locally: `tests/r2dreamer/test_agent.py`,
      `test_vggt_encoder.py`, `test_trainer.py`,
      `world_model/test_hybrid_encoder.py`, `test_habitat_act_state_parity.py`
- [ ] 4.6 Run the GPU-marked suite under `srun`/sbatch:
      `tests/r2dreamer/test_decoder_probe_overfit_gpu.py` and any
      Habitat/VGGT end-to-end checks that exercise `train_step` / `act` /
      `from_checkpoint`
- [ ] 4.7 Run `python -m pylint src/r2dreamer/agent.py` and the changed test
      paths before handoff