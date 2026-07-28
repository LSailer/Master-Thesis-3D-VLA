"""The token family: VGGT aggregator tokens, global, full-width, or pooled."""

from __future__ import annotations

import jax.numpy as jnp

from src.adapters.contract import (
    AdapterField,
    AdapterOutput,
    Encoder,
    FeatureExtractor,
)
from src.adapters.replay_image import replay_image
from src.environments.observation import ObservationFrame
from src.r2dreamer.encoders.constants import AGG_REGISTER_TOKENS
from src.vggt.jax.feature_extractor import VGGT_IMAGE_SIZE, VGGTExtractOutput

# Aggregator token layout along axis 0 (src/vggt/jax/aggregator.py): one camera
# token, then the register tokens, then one token per image patch. The register
# count is imported rather than restated: the pooled arm below slices past it
# while ``TokenTransformerEncoder`` drops it, and two copies drifting apart
# would silently pool the wrong tokens.
AGG_CAMERA_TOKEN_IDX = 0
AGG_PATCH_START_IDX = 1 + AGG_REGISTER_TOKENS


class GlobalTokensAdapter:
    """Routes the frame to a conv branch and the global tokens to a Transformer.

    The global half of the final aggregator tokens - all 1374 tokens of the last
    layer, 1024 channels each. Under streaming VGGT these carry house context
    accumulated in the attention cache, without any explicit point map, which is
    what this arm exists to measure. The point and camera heads are therefore
    switched off entirely.

    The tokens describe the *current* frame, so they are replayed per step rather
    than riding along as the live global field: a sampled window must see the
    tokens of the frames it holds, not the latest ones.

    Replay cost: 1374 x 1024 float16 is 2.8 MB per step, and the buffer
    preallocates ``capacity`` rows. Runs of this variant must cap
    ``--buffer_capacity`` accordingly - the default 500k rows would ask for
    petabytes.

    The token arms below vary only in ``TOKEN_KEY``, ``TOKEN_ENCODER``,
    ``WITH_RGB`` and :meth:`_tokens`; the extractor policy, the conv branch and
    the per-step replay decision are shared, so the arms stay comparable.
    """

    RENDER_RESOLUTION = VGGT_IMAGE_SIZE
    NEEDS_FEATURES = True
    # Heads off: every token arm reads aggregator tokens only, so the point and
    # camera heads would be pure cost.
    EXTRACTOR_KWARGS: dict[str, object] = {"compute_heads": False}
    ENCODER_OVERRIDES: dict[str, object] = {}

    TOKEN_KEY = "global_tokens"
    TOKEN_ENCODER = Encoder.TRANSFORMER
    WITH_RGB = True

    def __init__(self, extractor: FeatureExtractor) -> None:
        """Bind the frozen extractor this adapter reads tokens from."""
        self._extractor = extractor

    def _global_half(self, features: VGGTExtractOutput) -> jnp.ndarray:
        """Return the validated ``(num_tokens, token_dim)`` global token half."""
        tokens = jnp.asarray(features.global_tokens)
        if tokens.ndim != 2:
            raise ValueError(
                f"expected (num_tokens, token_dim) global tokens, got {tokens.shape}"
            )
        return tokens

    def _tokens(self, features: VGGTExtractOutput) -> jnp.ndarray:
        """Return the token payload this arm observes, in its replay dtype."""
        # float16, not the repo-default bfloat16: these are unit-scale
        # activations where the extra mantissa bits matter more than range, and
        # it halves an already large replay row.
        return self._global_half(features).astype(jnp.float16)

    def __call__(self, frame: ObservationFrame) -> AdapterOutput:
        """Route one env frame and its token context to the branches."""
        features = self._extractor.extract(frame)
        fields = [
            AdapterField(
                key=self.TOKEN_KEY,
                encoder=self.TOKEN_ENCODER,
                buffer=True,
                value=self._tokens(features),
            )
        ]
        if self.WITH_RGB:
            fields.append(
                AdapterField(
                    key="image",
                    encoder=Encoder.CONV,
                    buffer=True,
                    value=replay_image(frame.image),
                    decoder_target=True,
                )
            )
        return fields


class FullTokensAdapter(GlobalTokensAdapter):
    """Both halves of the aggregator tokens instead of the global one.

    The extractor splits the final full-width aggregator tokens into
    ``frame_tokens`` and ``global_tokens``; concatenating them back on the
    channel axis - in that order, as the legacy readout did - restores the
    ``(1374, 2048)`` block. This arm measures what the frame half adds over the
    global half alone.

    Replay cost doubles accordingly: 1374 x 2048 float16 is 5.6 MB per step, so
    ``--buffer_capacity`` has to be capped tighter than for the global arm.
    """

    TOKEN_KEY = "full_tokens"

    def _tokens(self, features: VGGTExtractOutput) -> jnp.ndarray:
        """Return the full-width tokens, frame half first."""
        return jnp.concatenate(
            [jnp.asarray(features.frame_tokens), self._global_half(features)], axis=-1
        ).astype(jnp.float16)


class AggregatorPooledAdapter(GlobalTokensAdapter):
    """The token sequence reduced to one ``(3072,)`` vector for an MLP branch.

    Pools the *global* half of the aggregator tokens - the same half the arms
    above read, which is what makes this the cheap end of the same comparison -
    into ``[camera token, patch mean, patch max]``. The register tokens are
    dropped: they carry no image content, they are an aggregator working set.

    No appearance channel and no geometry, so there is nothing to reconstruct
    and no decoder target. At 3072 float32 (12 KB per step) replay cost is
    negligible, unlike the sequence arms above.

    float32 rather than the token arms' float16: with only three pooled vectors
    left there is no replay pressure to trade precision against.

    The streaming KV cache runs on a 200 000 slot budget rather than the
    extractor default of 1 200 000 (50 000 per aggregator block). The large
    budget saturates after ~36 frames, and every step past that pays a
    full-cache eviction ``top_k`` per block; 200 000 slots shrink the temporal
    window to ~6 frames and halve the step cost. ``budgets_static`` stays
    unset, so the extractor derives the per-block cap from the total budget
    itself and the arm carries one number instead of a hand-written per-block
    split.

    Measured in Duell 2, see
    ``prototyp/duell-vggt-integration/2026-07-27-r2/LEDGER.md``: 134.1 -> 66.8
    ms per step in the 30-minute arena, at a matrix score of +0.0898 (seed 42)
    and +0.0194 (seed 43) over the 1.2M-budget reference run 6057641. The
    shorter context window is the explicit trade: reconstruction stays
    bit-identical up to cache saturation, and past it the deviation is
    dominated by a rigid shift plus a scale factor rather than by lost local
    geometry.
    """

    TOKEN_KEY = "agg_pooled"
    TOKEN_ENCODER = Encoder.MLP
    WITH_RGB = False
    EXTRACTOR_KWARGS: dict[str, object] = {
        "compute_heads": False,
        "total_budget": 200_000,
    }

    def _tokens(self, features: VGGTExtractOutput) -> jnp.ndarray:
        """Return ``[camera, patch mean, patch max]`` concatenated."""
        tokens = self._global_half(features)
        patches = tokens[AGG_PATCH_START_IDX:]
        return jnp.concatenate(
            [
                tokens[AGG_CAMERA_TOKEN_IDX],
                patches.mean(axis=0),
                patches.max(axis=0),
            ]
        ).astype(jnp.float32)


class AggregatorPooledBudget200kAdapter(AggregatorPooledAdapter):
    """Alias of :class:`AggregatorPooledAdapter`, kept for existing run ids.

    The 200 000 slot KV budget this class introduced is the pooled arm's
    default since the Duell 2 result, so the two are identical. The name stays
    registered as ``aggregator_pooled_b200k`` so the SLURM configs and the
    wandb run ids of the finished b200k runs keep resolving.
    """
