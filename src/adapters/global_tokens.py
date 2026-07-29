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


class AggregatorPooledFullAdapter(AggregatorPooledAdapter):
    """The pooled readout over the full-width tokens instead of the global half.

    Concatenates ``frame_tokens`` and ``global_tokens`` back to the ``(1374,
    2048)`` full-width block - frame half first, as the legacy readout did -
    and pools it exactly like the parent: ``[camera token, patch mean, patch
    max]``, now 3 x 2048 = 6144 float32 (24 KB per replay row). The camera
    token therefore carries both halves' camera tokens; the register tokens
    stay dropped.

    This is the one-variable comparison against the pooled arm: same MLP
    encoder, same KV budget, same pooling, only the frame half added. What the
    frame half carries that the global half does not is exactly what this arm
    measures (Duell 3, ``prototyp/duell-vggt-integration/2026-07-29-r3/``).
    """

    TOKEN_KEY = "agg_pooled_full"

    def _full_width(self, features: VGGTExtractOutput) -> jnp.ndarray:
        """Return the ``(num_tokens, 2048)`` full-width tokens, frame half first."""
        return jnp.concatenate(
            [jnp.asarray(features.frame_tokens), self._global_half(features)], axis=-1
        )

    def _tokens(self, features: VGGTExtractOutput) -> jnp.ndarray:
        """Return ``[camera, patch mean, patch max]`` over the full width."""
        tokens = self._full_width(features)
        patches = tokens[AGG_PATCH_START_IDX:]
        return jnp.concatenate(
            [
                tokens[AGG_CAMERA_TOKEN_IDX],
                patches.mean(axis=0),
                patches.max(axis=0),
            ]
        ).astype(jnp.float32)


class AggregatorPooledFrameMeanAdapter(AggregatorPooledFullAdapter):
    """The global pooled triple plus only the frame half's patch mean.

    ``[camera_g, patch mean_g, patch max_g, patch mean_f]`` = 4096 float32
    (16 KB per replay row). The cheap half-step between the pooled arm and
    :class:`AggregatorPooledFullAdapter`: it isolates whether the frame half's
    mean alone already carries the effect of the full-width readout.
    """

    TOKEN_KEY = "agg_pooled_meanf"

    def _tokens(self, features: VGGTExtractOutput) -> jnp.ndarray:
        """Return the global triple with the frame patch mean appended."""
        global_half = self._global_half(features)
        frame_half = jnp.asarray(features.frame_tokens)
        global_patches = global_half[AGG_PATCH_START_IDX:]
        frame_patches = frame_half[AGG_PATCH_START_IDX:]
        return jnp.concatenate(
            [
                global_half[AGG_CAMERA_TOKEN_IDX],
                global_patches.mean(axis=0),
                global_patches.max(axis=0),
                frame_patches.mean(axis=0),
            ]
        ).astype(jnp.float32)


class AggregatorPooledFullDeltaAdapter(AggregatorPooledFullAdapter):
    """The full-width pooled readout plus a camera-token delta to frame 0.

    Appends ``camera_t - camera_0`` as a fourth 2048 block (8192 float32,
    32 KB per replay row), where ``camera_0`` is the camera token of the
    episode's first frame. The delta is the latent stand-in for relative pose:
    frame 0 is the episode's cache anchor, so the difference encodes where the
    agent has moved since the start without the camera head being computed.

    The anchor is per-episode state on the adapter, reset on ``is_first`` -
    mirroring the extractor's own episode-reset boundary, so anchor and cache
    always restart together.
    """

    TOKEN_KEY = "agg_pooled_full_delta"

    def __init__(self, extractor: FeatureExtractor) -> None:
        """Bind the extractor and start with no episode anchor."""
        super().__init__(extractor)
        self._camera_anchor: jnp.ndarray | None = None

    def __call__(self, frame: ObservationFrame) -> AdapterOutput:
        """Drop the anchor on episode start, then route as usual."""
        if frame.is_first:
            self._camera_anchor = None
        return super().__call__(frame)

    def _tokens(self, features: VGGTExtractOutput) -> jnp.ndarray:
        """Return the full-width triple plus the camera delta block."""
        tokens = self._full_width(features)
        camera = tokens[AGG_CAMERA_TOKEN_IDX]
        if self._camera_anchor is None:
            self._camera_anchor = camera
        patches = tokens[AGG_PATCH_START_IDX:]
        return jnp.concatenate(
            [
                camera,
                patches.mean(axis=0),
                patches.max(axis=0),
                camera - self._camera_anchor,
            ]
        ).astype(jnp.float32)


class AggregatorPooledFullSplitAdapter(AggregatorPooledFullAdapter):
    """The full-width pooled blocks as three routed fields, one MLP branch each.

    Same information as :class:`AggregatorPooledFullAdapter`, but camera token,
    patch mean and patch max are separate 2048 fields. Each gets its own MLP
    branch and the composite encoder fuses the three embeddings, instead of one
    6144 -> 1024 projection mixing the blocks in a single layer. Replay cost is
    identical to the single-field arm (3 x 8 KB per row).
    """

    def __call__(self, frame: ObservationFrame) -> AdapterOutput:
        """Route the three pooled blocks as separate MLP fields."""
        features = self._extractor.extract(frame)
        tokens = self._full_width(features)
        patches = tokens[AGG_PATCH_START_IDX:]
        blocks = (
            ("agg_camera", tokens[AGG_CAMERA_TOKEN_IDX]),
            ("agg_patch_mean", patches.mean(axis=0)),
            ("agg_patch_max", patches.max(axis=0)),
        )
        return [
            AdapterField(
                key=key,
                encoder=Encoder.MLP,
                buffer=True,
                value=value.astype(jnp.float32),
            )
            for key, value in blocks
        ]


class AggregatorPooledFullQuadAdapter(AggregatorPooledFullAdapter):
    """The full-width pooled readout plus 2x2 spatial quadrant means.

    Appends the mean of each quadrant of the 37 x 37 patch grid to the P1
    triple: ``[camera, mean, max, q00, q01, q10, q11]`` = 7 x 2048 = 14336
    float32 (56 KB per replay row, 28 GB at 500k capacity). mean/max discard
    all spatial structure; four quadrant means are the cheapest step that
    restores any - roughly "what is left/right/above/below of me" at zero
    VGGT cost.
    """

    TOKEN_KEY = "agg_pooled_full_quad"
    # 1369 patches = 37 x 37, the VGGT patch grid at 518 px / patch 14.
    PATCH_GRID = 37

    def _tokens(self, features: VGGTExtractOutput) -> jnp.ndarray:
        """Return the P1 triple plus the four quadrant means."""
        tokens = self._full_width(features)
        patches = tokens[AGG_PATCH_START_IDX:]
        grid = patches.reshape(self.PATCH_GRID, self.PATCH_GRID, -1)
        half = self.PATCH_GRID // 2
        quadrants = [
            grid[:half, :half],
            grid[:half, half:],
            grid[half:, :half],
            grid[half:, half:],
        ]
        return jnp.concatenate(
            [
                tokens[AGG_CAMERA_TOKEN_IDX],
                patches.mean(axis=0),
                patches.max(axis=0),
                *[q.mean(axis=(0, 1)) for q in quadrants],
            ]
        ).astype(jnp.float32)


class AggregatorPooledFrameOnlyAdapter(AggregatorPooledFullAdapter):
    """The pooled triple over the frame half alone.

    ``[camera_f, patch mean_f, patch max_f]`` = 3072 float32 (12 KB per replay
    row) - the exact mirror of the global-half pooled arm. Against P1 and the
    pooled arm it separates "the frame half adds information" from "the frame
    half is where the information lives".
    """

    TOKEN_KEY = "agg_pooled_frame"

    def _tokens(self, features: VGGTExtractOutput) -> jnp.ndarray:
        """Return ``[camera, patch mean, patch max]`` over the frame half."""
        tokens = jnp.asarray(features.frame_tokens)
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
