## ADDED Requirements

### Requirement: `R2DreamerAgent.__init__` produces a parameter-less shell
`R2DreamerAgent.__init__` SHALL build the Flax module objects, the module
dict, the optimizer builder, the return-EMA, the acting state, and the JIT
train/act wrappers, and SHALL NOT materialise the parameter pytree, the
optimizer state, the slow-target critic params, the EMA state, or
`embed_size`. It SHALL stash the construction RNG for later use by
`initialize` and mark the agent as not-yet-initialised.

#### Scenario: Agent is not initialised immediately after construction
- **WHEN** an `R2DreamerAgent` is constructed with a config and RNG key
- **THEN** `agent.params`, `agent.opt_state`, `agent.slow_critic_params`,
  and `agent.embed_size` are not yet available, and the agent is marked
  not-initialised

#### Scenario: Module objects are available before initialisation
- **WHEN** a caller reads `agent._modules` before the first step
- **THEN** the module-object dict is populated (it does not depend on
  parameters), so downstream readers such as held-out evaluation continue to
  work

### Requirement: `initialize` materialises parameters from a real observation
`R2DreamerAgent.initialize(seed_obs, *, params_override=None)` SHALL split the
stashed construction RNG, initialise the encoder/RSSM/projector/heads/decoder,
compute `embed_size` from `encoder_mod.apply` on the provided `seed_obs`, build
the optimizer state and slow-target critic params, assemble the training
state, and mark the agent initialised — exactly once. When `params_override`
is supplied it SHALL adopt those parameters in place of the freshly
initialised ones.

#### Scenario: First real batch initialises the agent
- **WHEN** `train_step` (or `act` / `act_with_state`) is called on an
  not-yet-initialised agent
- **THEN** the agent auto-initialises from the real observation carried by
  that call (`batch.obs` for training, the batched live observation for
  acting) before performing the step

#### Scenario: `embed_size` comes from the real observation
- **WHEN** `initialize` runs with a real `seed_obs`
- **THEN** `embed_size` equals `encoder_mod.apply(enc_params,
  seed_obs).shape[-1]`, with no synthesised dummy observation on the
  training/acting path

#### Scenario: `from_checkpoint` returns a ready, checkpoint-weighted agent
- **WHEN** `R2DreamerAgent.from_checkpoint(path, ...)` is called
- **THEN** it builds the shell, runs `initialize` with a shape-only zeros
  observation and `params_override` set to the checkpoint's params, assigns
  the checkpoint's slow-target critic params, and returns an agent whose
  parameters are the checkpoint's — without discarding a freshly initialised
  pytree

#### Scenario: Double initialisation is rejected
- **WHEN** `initialize` is called on an agent that is already initialised
- **THEN** it raises rather than silently regenerating parameters and
  discarding the existing training state

### Requirement: Lazy initialisation preserves parameter structure and numerics
The parameter pytree materialised by `initialize` SHALL be structurally
identical (same leaves, shapes, and dtypes) to the pytree the pre-change eager
`__init__` produced for the same config and construction RNG, and runtime
train/act/eval numerics SHALL be unchanged. The change is a timing refactor
only.

#### Scenario: Param structure matches the eager baseline
- **WHEN** an agent is initialised lazily with a given config and RNG key
- **THEN** its parameter pytree matches, leaf-for-leaf in shape and dtype,
  the pytree the eager `__init__` produced for the same config and key, and
  values match for the same key

#### Scenario: Numerics are unchanged across the encoder_type matrix
- **WHEN** `train_step`, `act`, and `from_checkpoint` are exercised across the
  supported `encoder_type` values
- **THEN** the existing test suite (agent, vggt encoder, trainer, hybrid
  encoder, habitat act-state parity, decoder probe) passes without numeric
  regressions