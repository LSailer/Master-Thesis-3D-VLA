# Open problems and caveats

- **Vendored code dominates the ranking.** Six of the eight hotspots are the
  JAX VGGT port under `src/vggt/jax/`, which is a transcription of an upstream
  PyTorch model. Its complexity is inherited, not authored, and its high blast
  radius is an artefact of a deep single-purpose import chain
  (`attention <- block <- aggregator <- feature_extractor`). Deciding whether
  to exclude `src/vggt/` is a judgement call, not a bug.

- **Blast radius counts modules, not calls.** A module importing another for a
  single constant weighs the same as one built entirely on top of it. Weighting
  edges by the number of imported names would sharpen this.

- **Static imports only.** Dynamic imports, `importlib` and plugin-style
  lookups are invisible to the AST pass. The repo uses absolute `src.x.y`
  imports throughout, so this is currently not a practical gap.

- **Cyclomatic complexity is a path count, not a readability measure.**
  `complexipy` rates `run_loop` at 82 cognitive against 31 cyclomatic, because
  nesting is weighted. Adding cognitive complexity as a third axis is the
  obvious next step; complexipy is a Rust CLI, so it would have to be shelled
  out to and parsed rather than imported.

- **Git churn is missing.** The classic hotspot formula is complexity x change
  frequency (Tornhill). Blast radius is a structural proxy for the same idea.
  Adding `git log --numstat` churn per module would give the empirical version.

## LCOM4 false positives, and what removing them cost

Textbook LCOM4 flagged 8 classes here. Every one was an artefact, and the
filters had to be added one at a time against real examples:

| Construct | Why it is isolated by construction | Example |
|---|---|---|
| `Protocol` / ABC / stub bodies | methods share no state on purpose | `ExperienceSource` scored LCOM4 9 of 9 methods |
| `@classmethod` | binds `cls`, never `self` | `R2DreamerAgent.from_checkpoint` |
| constant-returning properties | read no instance state at all | `JAXVGGTFeatureExtractor.image_size`, `HabitatObjectNavEnv.num_actions` |
| `__init__` | touches every attribute, fuses all components | would drive nearly every LCOM4 to 1 |

After filtering, one finding survives repo-wide, and it is cosmetic
(`ExperienceCollector.diagnostics`, a one-line pass-through to an injected
`diagnostics_fn` that nothing else reads). The honest conclusion is that LCOM4
has no purchase on this codebase; CBO is the class-level metric that carries
information here.

- **CBO matches on bare names.** Two project classes sharing a name are
  conflated, and a class referenced only in a type annotation weighs the same
  as one that is instantiated and driven. Resolving imports per module would
  fix the first, weighting by reference kind the second.

- **Flax modules distort WMC and DIT.** `nn.Module` subclasses declare
  attributes as class-level annotations and implement in `setup`/`__call__`,
  so their method graph looks different from a plain Python class. DIT counts
  project bases only, so every Flax class reads DIT 0.
