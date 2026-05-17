# Pose as JAX leaf — design proposal for B1

Status: **proposal** | Branch: `feat/encoder-fusion-ablation` | Resolves: blocker B1 of [encoder-fusion-plan-audit.md](encoder-fusion-plan-audit.md)

## Context

The [encoder-fusion plan](encoder-fusion-plan.md) Phase 3a wants to log a `pose_grad_norm` diagnostic computed as `jax.grad(lambda p: stoch_logits(... pose=p ...).sum())(pose)`. The audit caught that this snippet cannot run today, because pose is bundled inside a flat `obs` array by the time JAX sees it. This doc proposes the smallest API change that makes pose a separable JAX leaf inside the loss function, without touching the replay buffer schema.

## Current data flow

1. The Habitat wrapper produces `obs["image"]` plus the new `obs["agent_state"]` (after B2).
2. The VGGT acting path packs `world_points (4107) + camera_pose (9)` into a flat 4116-D vector via [_flatten_vggt at vggt_adapter.py:14](../../../modules/r2dreamer/adapters/vggt_adapter.py#L14). This array goes into the replay buffer as `obs`.
3. The trainer samples a batch and calls [_loss_fn at agent.py:359](../../../modules/r2dreamer/agent.py#L359). At line 373 it reshapes `batch["obs"]` to `(B*T, 4116)` and passes it to the encoder: `embed = self.encoder_mod.apply(params["encoder"], obs_flat)`.
4. Inside [VGGTEncoder.__call__ at networks.py:414](../../../modules/r2dreamer/networks.py#L414): `out = nn.Dense(self.embed_dim, name="proj")(obs)`. Pose enters as the last 9 dims of the flat input. There is no separable handle to it for `jax.grad`.

## Proposed restructure

Make `pose` (and any other encoder-relevant non-image input) a **named JAX argument** alongside `obs`, sliced ONCE inside `_loss_fn` before the encoder call. The replay buffer schema stays untouched.

### API change

Replace the call site in [_loss_fn at agent.py:374](../../../modules/r2dreamer/agent.py#L374):

```python
# before
embed = self.encoder_mod.apply(params["encoder"], obs_flat)

# after
obs_flat, pose_flat = self._split_obs_pose(batch["obs"], B, T)
embed = self.encoder_mod.apply(params["encoder"], obs_flat, pose_flat)
```

Add a tiny helper `_split_obs_pose(obs_packed, B, T)` on the agent that handles the slicing per encoder family (cheap dispatch on `self.cfg.encoder_name`), keeps the replay-buffer flat layout intact, and is jit-friendly.

Update every encoder `__call__` to accept the new positional `pose` arg:

| Encoder | New signature | Internal change |
|---|---|---|
| `ConvEncoder` (CNN baseline, no pose) | `__call__(self, obs, pose=None)` | Ignore `pose` |
| `VGGTEncoder` (current baseline) | `__call__(self, obs, pose)` | Concat internally — numerically identical to the existing single-Dense path |
| `VGGTEncoder` for `vggt_pose_scaled` | same, with `pose * 100` scaling | Same |
| `TileEncoder` | `__call__(self, obs, pose)` | Already needed pose as input — minor refactor |
| `FiLMEncoder` | `__call__(self, obs, pose)` | Pass `pose` to FiLM head |
| `PluckerEncoder` | `__call__(self, obs, pose, extr, intr)` | Add explicit args for the Habitat extrinsics + intrinsics surfaced by B2 |
| `CrossAttnEncoder` | `__call__(self, obs, pose)` | Pose enters via the query MLP — already needed |

### Why the buffer stays untouched

The replay buffer's flat 4116-D layout (or 91732-D after Phase 1) is preserved end-to-end. Only the *post-sample, pre-encoder* slicing changes — done inside the JIT-compiled `_loss_fn`, so it's free at runtime. This decouples B1 (encoder API) from B3 (buffer storage), letting them ship independently.

### Diagnostic enabled

After the restructure, the Phase 3a snippet works:

```python
def stoch_logits_from_pose(p):
    embed = self.encoder_mod.apply(params["encoder"], obs_flat, p)
    post, _ = self.rssm_mod.apply(params["rssm"], stoch, deter, action, embed)
    return post["logits"].sum()

grad_pose = jax.grad(stoch_logits_from_pose)(pose_flat)
metrics["pose_grad_norm"] = jnp.linalg.norm(grad_pose)
```

One extra `grad` per train step. Cheap inside the existing JIT.

## Cross-variant interpretation

`pose_grad_norm` is comparable across variants in the **directional** sense ("does pose enter the loss at all"), but the **absolute** magnitudes are not on the same scale:

- For `vggt` / `vggt_pose_scaled` (linear path): JVP magnitude through one Dense.
- For `vggt_tile` / `vggt_plucker` (channel concat): gradient flux through Conv kernels.
- For `vggt_film`: gradient through the γ/β MLP, then composition with the modulation operation.
- For `vggt_xattn`: gradient through the query MLP, then through softmax attention.

The results page must report `pose_grad_norm` per variant **and** as a ratio to the baseline `vggt`. The ratio is what decides "did this variant break the pose-blockage."

## Backwards compatibility

What breaks:
- Every encoder unit test that calls `encoder(obs)` directly.
- The smoke train hook in [registries.py:15](../../../modules/r2dreamer/launch/registries.py#L15) (encoder dispatch — needs the new arg).
- Any standalone parity script that exercises the encoder.

What stays:
- Replay buffer schema (untouched).
- Acting path (the adapter's `_flatten_vggt` still produces the same flat bytes).
- RSSM contract (`embed (B, embed_dim)` is unchanged).
- The 75% SR baseline numerical output (the new `VGGTEncoder` path concatenates `obs_flat[:, :-9]` and `pose_flat` then applies the same Dense — bit-identical).

## Estimated diff size

| Files | LOC (net) |
|---|---|
| [modules/r2dreamer/agent.py](../../../modules/r2dreamer/agent.py) — add `_split_obs_pose`, update call site | +30 |
| [modules/r2dreamer/networks.py](../../../modules/r2dreamer/networks.py) — update existing `VGGTEncoder` + `ConvEncoder` signatures | +15 |
| [modules/r2dreamer/launch/registries.py](../../../modules/r2dreamer/launch/registries.py) — none (registry is just class refs) | 0 |
| `tests/` — new encoder-API tests + parity check that baseline `vggt` is bit-identical before/after | +60 |
| **Total** | **~100 LOC** across 3 files + 1 new test file |

## Testing strategy

1. **Encoder unit tests** (new): for each encoder, assert `embed = encoder(obs, pose)` produces the right shape, and that `jax.grad(lambda p: encoder(obs, p).sum())(pose)` returns a finite array of shape `pose.shape`.
2. **Baseline parity** (new): build the current `VGGTEncoder` with old API and the new one, run on the same random input, assert `jnp.allclose(out_old, out_new, atol=1e-6)`. This protects the 75% SR baseline from accidental regressions.
3. **Smoke train**: 1k env steps with `--encoder vggt`. SR/loss curve matches the act_entropy=3e-2 baseline within noise.
4. **`pose_grad_norm` smoke**: confirm the diagnostic returns a finite scalar at init; for the baseline `vggt` it should be small but non-zero (linear path is non-degenerate); for `cnn` it should be exactly zero (encoder ignores pose).

## Decision needed before implementation

- Confirm the new `pose` argument should be **positional after `obs`** (this doc's choice) vs. a keyword-only kwarg. Positional is simpler for Flax `nn.Module` and matches the pattern used by RSSM.
- Confirm `_split_obs_pose` belongs on the agent (this doc) vs. inside the adapter. Agent-side keeps the buffer untouched and the slicing JIT-compiled.

If both are accepted, this is a one-PR change.
