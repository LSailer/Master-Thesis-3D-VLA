"""Encoder specs: launcher-side description of the full input pipeline.

Each subclass of `Encoder` is the single source of truth for one encoder choice:
the env-side adapter, the env render resolution, agent config overrides, AND the
matching Flax `nn.Module` class consumed inside the agent. The agent reads
`module_cls` from the resolved `EncoderSpec` and instantiates the network on
its side of the `jax.jit` boundary — the launcher never holds a live module
instance.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import flax.linen as nn

from src.r2dreamer.adapters.obs_adapter import ObsAdapter
from src.r2dreamer.adapters.vggt_adapter import VGGTObsAdapter
from src.r2dreamer.world_model import encoders as wm_encoders
from src.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor as VGGTFeatureExtractor


@dataclass(frozen=True)
class EncoderSpec:
    """Full description of an encoder choice, including its `nn.Module` class.

    `module_cls` is the Flax module the agent will instantiate (with kwargs
    pulled from `R2DreamerConfig`) — passing the class object rather than an
    instance keeps the launcher on the non-JIT side of the boundary.
    """

    obs_shape: tuple[int, ...]
    env_render_resolution: int
    encoder_type: str
    module_cls: type[nn.Module]
    agent_overrides: dict[str, Any] = field(default_factory=dict)
    design_notes: str = ""


class Encoder(ABC):
    """Base class for everything an agent might consume as input.

    Subclasses declare *what they are* via class attributes
    (encoder_type / module_cls / agent_overrides / design_notes /
    env_render_resolution) and *how to build their adapter* via _build_adapter().
    The base spec() sources obs_shape from the adapter so encoder and adapter
    can never disagree on observation shape.
    """

    encoder_type: str = ""
    module_cls: type[nn.Module] | None = None
    env_render_resolution: int = 64
    # Class-attribute defaults are read-only here: subclasses *reassign*
    # them (e.g. agent_overrides = {...}) and spec() copies via dict(...)
    # before exposure, so the shared instance is never mutated in place.
    agent_overrides: dict[str, Any] = {}
    design_notes: str = ""

    _adapter: ObsAdapter | None = None

    @classmethod
    def from_train_args(cls, args: Any) -> "Encoder":
        """Construct an encoder from train() CLI args."""
        return cls()

    def make_adapter(self) -> ObsAdapter:
        """Return the ObsAdapter that bridges env obs to agent input (cached)."""
        if self._adapter is None:
            self._adapter = self._build_adapter()
        return self._adapter

    @abstractmethod
    def _build_adapter(self) -> ObsAdapter:
        """Build the adapter. Called at most once per encoder; result is cached."""

    def spec(self) -> EncoderSpec:
        """Static observation/env/agent requirements for train()."""
        if self.module_cls is None:
            raise NotImplementedError(
                f"{type(self).__name__} must set module_cls (a Flax nn.Module class)"
            )
        adapter = self.make_adapter()
        return EncoderSpec(
            obs_shape=adapter.buffer_shape,
            env_render_resolution=self.env_render_resolution,
            encoder_type=self.encoder_type,
            module_cls=self.module_cls,
            agent_overrides=dict(self.agent_overrides),
            design_notes=self.design_notes,
        )


class CNNEncoder(Encoder):
    """Identity encoder — agent's internal CNN handles RGB -> embedding -> RSSM."""

    encoder_type = "cnn"
    module_cls = wm_encoders.ConvEncoder

    def _build_adapter(self) -> ObsAdapter:
        return ObsAdapter()  # passthrough, default behavior


class VGGTEncoder(Encoder):
    """External feature extractor — 518x518 RGB -> 4116-dim flat vector."""

    # Match the fast JAX benchmark configuration (`--jax-static-budgets`).
    # Dynamic per-layer budgets trigger JAX/XLA recompilation after cache
    # eviction starts because the budget tuple is a jit static argument.
    VGGT_TOTAL_BUDGET = 200_000
    VGGT_STATIC_BUDGETS = tuple([8333] * 24)

    feature_kind = "wp_cp"
    encoder_type = "vggt"
    module_cls = wm_encoders.VGGTEncoder
    agent_overrides = {"buffer_capacity": 1_000_000}
    # Subclasses set vggt_compute_heads = False when they consume only the
    # pre-head aggregator tokens, so the extractor can skip camera_head +
    # point_head + world_points wrapper on every frame.
    vggt_compute_heads = True
    # Grid the dense point map is pooled to for the WP output (37 = patch grid).
    wp_pool_size = 37

    @classmethod
    def from_train_args(cls, args: Any) -> "VGGTEncoder":
        return cls(resolution=args.render_resolution)

    def __init__(self, resolution: int = 518):
        self.env_render_resolution = resolution
        self._extractor = VGGTFeatureExtractor(
            total_budget=self.VGGT_TOTAL_BUDGET,
            budgets_static=self.VGGT_STATIC_BUDGETS,
            compute_heads=self.vggt_compute_heads,
            wp_pool_size=self.wp_pool_size,
        )  # device="cuda" default

    def _build_adapter(self) -> ObsAdapter:
        return VGGTObsAdapter(self._extractor, feature_kind=self.feature_kind)


class VGGTAggregatorMLPEncoder(VGGTEncoder):
    """External VGGT extractor exposing pooled pre-head aggregator features."""

    feature_kind = "aggregator"
    encoder_type = "vggt_aggregator_mlp"
    module_cls = wm_encoders.VGGTAggregatorMLPEncoder
    # Aggregator-only path: skip camera_head + point_head + world_points wrapper
    # in the extractor; only `aggregator_features` is needed downstream.
    vggt_compute_heads = False
    agent_overrides = {
        "buffer_capacity": 5_000,
        "batch_size": 4,
        "seq_len": 32,
        "train_ratio": 128,
    }
    design_notes = (
        "Variant 1 encoder: VGGT final pre-head global aggregator tokens "
        "(1374x1024 = 1 camera token + 4 register tokens + 37x37 patch tokens) "
        "are pooled adapter-side into three 1024-dim vectors before replay: "
        "(a) the camera token (idx 0) is kept unmixed because VGGT's own "
        "camera_head reads it for pose; (b) mean over patch tokens (idx 5:) "
        "is a smooth global summary; (c) max over the same patches surfaces "
        "salient features. The three are concatenated to a flat (3072,) "
        "vector, stored as float32 in replay, and the encoder applies a "
        "per-slice RMSNorm followed by a 2-layer MLP -> embed_dim. The "
        "camera-pose head is skipped (vggt_compute_heads=False) since the "
        "camera-token embedding itself already carries pose information."
    )


class VGGTDenseWPEncoder(VGGTEncoder):
    """Full-resolution world-point map (518x518x3) -> Conv encoder (3D-53).

    Unlike the WP/CP variant, this skips the 14x14 average-pool to 37x37 and
    feeds the dense per-pixel point map straight into a conv stack that treats
    the XYZ coordinates as a 3-channel image. The point head still runs
    (``vggt_compute_heads=True``) because the dense map is its raw output;
    only the pooling + flatten+Dense readout is dropped.
    """

    feature_kind = "wp_dense"
    encoder_type = "vggt_wp_dense_cnn"
    module_cls = wm_encoders.WPConvEncoder
    # Dense WP needs the point head (the 518² map is its pre-pool output).
    vggt_compute_heads = True
    # A 518²x3 float16 frame is ~1.6 MB, vs ~16 KB at 37². Keep the buffer small
    # and batches modest so replay storage and per-step conv cost stay bounded.
    agent_overrides = {
        "buffer_capacity": 5_000,
        "batch_size": 4,
        "seq_len": 32,
        "train_ratio": 128,
    }
    design_notes = (
        "Variant: full-resolution VGGT world points. The DPT point head's dense "
        "518x518x3 per-pixel map (one metric XYZ point per pixel) is NOT pooled "
        "to 37x37; it is stored channel-first as a (3, 518, 518) float16 image "
        "and fed to WPConvEncoder, which symlog-normalises the metric XYZ range "
        "and runs the RGB Conv+MaxPool+RMSNorm+SiLU stack before a linear "
        "readout to embed_dim. WP-only (no camera pose): a 9-vector cannot be a "
        "spatial channel. See issue 3D-53."
    )


class HybridEncoder(VGGTEncoder):
    """CNN(RGB 64) + gated MLP(WP+CP 4116) fused into a single latent."""

    feature_kind = "wp_cp"
    encoder_type = "hybrid"
    module_cls = wm_encoders.HybridEncoder
    # World_points + camera_pose are required, so the extractor must build at
    # 518 with heads (compute_heads=True, inherited from VGGTEncoder).
    vggt_compute_heads = True
    # 16404-d float32 at 1M ~= 65GB host RAM > node's 64GB; 100k ~= 6.5GB.
    agent_overrides = {"buffer_capacity": 100_000}
    design_notes = (
        "Hybrid encoder: a CNN branch over the 64x64 RGB frame and a gated MLP "
        "branch over the 4116-dim VGGT world_points+camera_pose vector are fused "
        "(concatenated) into the latent. A zero-init scalar gate on the WP/CP "
        "branch means training starts as plain CNN-Dreamer and only blends in the "
        "geometric features as the gate opens; per-branch contributions are logged "
        "as hybrid/* metrics (3D-50/51/52)."
    )

    def _build_adapter(self) -> ObsAdapter:
        from src.r2dreamer.adapters.hybrid_adapter import HybridObsAdapter

        return HybridObsAdapter(self._extractor)


class HybridNormFixedEncoder(HybridEncoder):
    """CNN(RGB 64) + fixed-scale normalized MLP(WP+CP) ablation (3D-65)."""

    encoder_type = "hybrid_norm_fixed"
    module_cls = wm_encoders.HybridNormFixedEncoder
    agent_overrides = {
        "buffer_capacity": 100_000,
        "hybrid_fixed_scale": 1.0,
    }
    design_notes = (
        "3D-65 hybrid ablation: same input layout as the default hybrid "
        "([64x64 RGB | 4116-dim VGGT world_points+camera_pose]) but both branch "
        "embeddings are parameter-free RMS-normalized before concatenation. The "
        "VGGT branch uses a fixed scale of 1.0 instead of a learned zero-init gate, "
        "so the run tests whether the learned gate caused bad branch-scale dynamics."
    )


class VGGTWPCP64Encoder(VGGTEncoder):
    """WP+CP MLP at a finer 64x64 world-point grid (3D-52/3D-53 follow-up).

    Identical to the WP+CP MLP encoder (same MLP module, same camera pose, same
    1M-transition replay buffer) but pools VGGT's dense 518x518 point map to
    64x64 instead of 37x37: obs becomes 64*64*3 + 9 = 12297-D. Holding the
    architecture and pose fixed and sweeping only the WP resolution (37 -> 64)
    isolates whether finer geometry helps; 64x64 also matches the RGB-CNN
    baseline's input resolution. Replaces the conv-on-518² path (WPConvEncoder),
    since a 12297-D vector is a trivial MLP input (~0.2 ms encoder forward) while
    the 518² conv was ~4x too slow to finish.
    """

    encoder_type = "vggt_wp_cp_64"
    wp_pool_size = 64
    # Inherits feature_kind="wp_cp", module_cls=VGGTEncoder (MLP),
    # vggt_compute_heads=True, agent_overrides={buffer_capacity: 1_000_000}.
    # 64²x3 float32 ~= 49 KB/frame, so 1M transitions ~= 49 GB host RAM -> the
    # run config bumps node memory accordingly.
    design_notes = (
        "WP+CP MLP at a 64x64 world-point grid. VGGT's dense 518x518x3 point map "
        "is average-pooled (antialiased area resample, since 518 is not divisible "
        "by 64) to 64x64x3, flattened (12288) and concatenated with the 9-D camera "
        "pose into a 12297-D observation, then encoded by the same multi-layer MLP "
        "as the 37x37 WP+CP variant. Only the WP resolution differs (37 -> 64): a "
        "controlled resolution ablation, at the RGB-CNN baseline resolution."
    )


__all__ = [
    "EncoderSpec",
    "Encoder",
    "CNNEncoder",
    "VGGTEncoder",
    "VGGTAggregatorMLPEncoder",
    "VGGTDenseWPEncoder",
    "HybridEncoder",
    "HybridNormFixedEncoder",
    "VGGTWPCP64Encoder",
]
