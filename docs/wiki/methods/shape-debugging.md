# Shape debugging — JAX/Flax + PyTorch

Reference for tracing tensor dimensions through R2-Dreamer / DreamerV3 forward passes. Read this when a shape mismatch fires.

## Where dimension constants live

- `src/dreamerv3/configs.py` — DreamerV3 dims (`obs_shape`, `hidden_size`, `latent_classes`, `latent_dims`)
- `src/r2dreamer/config.py` — R2-Dreamer dims (`deter_size`, `stoch_classes`, `stoch_discrete`, `vggt_embed_dim`)
- `src/dreamerv3/networks.py` — JAX/Flax encoder, decoder, RSSM, heads
- `src/r2dreamer/networks.py` — R2-Dreamer JAX networks (extends DreamerV3)
- `src/vggt/` — VGGT feature extractor (output dim feeds encoder fusion points)

## Tracing pattern

1. Start at the config to pin dimension constants.
2. Walk each module's `__call__` / `__init__`, tracking transforms through:
   - Conv layers (kernel, stride, padding → spatial dims)
   - Linear/Dense (`in_features → out_features`)
   - Reshape / transpose / flatten
   - `jnp.concatenate` / `torch.cat` (axis alignment)
   - `einsum` (index matching)
   - `jax.nn.one_hot` (adds trailing dim)
3. Verify boundaries: encoder out → RSSM in; RSSM state → decoder in; RSSM state → reward/continue heads; VGGT features → encoder fusion points.
4. Confirm batch-dim convention: JAX often vmaps over batch (no batch dim inside); PyTorch carries batch dim. Mixing the two is a frequent source of confusion.

## Known footguns

- **Stoch size formula differs across frameworks.** R2 uses `stoch_classes × stoch_discrete`; D3 uses `latent_classes × latent_dims`. These are separate configs — don't reuse one for the other.
- **RSSM state concatenation.** `[deter, stoch]` along feature dim must add up to `feat_size`.
- **Conv spatial arithmetic.** With 4 conv layers at stride 2: `64×64 → 4×4`; `256×256 → 16×16`. Check before/after each conv block.
- **NCHW vs NHWC.** JAX Flax convs default to NHWC. Observations are stored NCHW. Look for the transpose; missing it produces silent broadcasts or hard errors much later.
- **TwoHot bins.** R2-Dreamer uses 255 real-space bins; DreamerV3 uses 255 symlog bins. Logits dimensions must match the chosen scheme.

## Reporting format (when documenting a hunt in a lessons file)

```
## <filename>

### Shapes traced
- <ModuleName>.__call__: input (...) → output (...)

### Issues found
- [MISMATCH] file:line — expected (B, 512), got (B, 256) because <reason>
- [WARNING] file:line — implicit reshape, verify intent

### Clean
- <list of interfaces verified as correct>
```

A clean report — "traced X interfaces, no mismatches" — is itself a valuable artifact.
