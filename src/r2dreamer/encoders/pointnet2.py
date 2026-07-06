"""Skeleton PointNet++ encoder family for the external ``pointnet2`` repo.

The downloaded ``external/pointnet2`` project is TensorFlow 1 code. This module
does not port or call that implementation; it only documents the R2Dreamer seam
where a future adapter can turn point-cloud observations into RSSM embeddings.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, NamedTuple

import flax.linen as nn
import jax

from src.r2dreamer.encoders.base import Encoder
from src.r2dreamer.encoders.constants import HOUSE_CONTEXT_MAX_POINTS, HOUSE_POINT_DIM
from src.r2dreamer.observation_keys import HOUSE_CONTEXT_KEY

PointNet2Variant = Literal["ssg", "msg"]


class PointNet2PipelineSpec(NamedTuple):
    """Static contract sketch for the eventual PointNet++ pipeline wiring."""

    external_repo_path: str = "external/pointnet2"
    encoder_type: str = "pointnet2"
    input_key: str = HOUSE_CONTEXT_KEY
    point_shape: tuple[int, int] = (HOUSE_CONTEXT_MAX_POINTS, HOUSE_POINT_DIM)
    point_dtype: str = "float32"
    encoder_output_shape: tuple[int, ...] = (1024,)
    encoder_output_dtype: str = "float32"
    module_class_name: str = "PointNet2FeatureEncoder"
    adapter_class_name: str = "PointNet2ObsAdapter"


class PointNet2BackboneOutput(NamedTuple):
    """Output expected from the future PointNet++ backbone adapter."""

    embedding: jax.Array
    endpoints: Mapping[str, jax.Array]


class PointNet2FeatureEncoder(nn.Module):
    """Flax encoder-module skeleton for PointNet++ point-cloud features.

    Initializer fields:
        external_repo_path: Relative path to the downloaded TensorFlow PointNet++
            repo, currently ``external/pointnet2``.
        variant: External classifier family, ``"ssg"`` or ``"msg"``.
        num_points: Static point count expected by JAX/JIT.
        point_dim: Per-point feature width. The current house-point pipeline uses
            ``xyzrgb`` with width 6; the original external classifier consumes
            ``xyz`` with width 3.
        embed_dim: Width of the RSSM encoder embedding.
        input_key: Observation mapping key containing the point-cloud tensor.

    Public functions:
        ``__call__(obs, train=False)`` expects ``obs[input_key]`` with shape
        ``(..., num_points, point_dim)`` and returns a float array with shape
        ``(..., embed_dim)`` for the RSSM posterior.
        ``encode_points(points, train=False)`` is the future PointNet++ seam. It
        expects ``points`` with shape ``(..., num_points, point_dim)`` and returns
        ``PointNet2BackboneOutput`` with the final embedding plus named
        intermediate endpoint tensors.
    """

    external_repo_path: str = "external/pointnet2"
    variant: PointNet2Variant = "ssg"
    num_points: int = HOUSE_CONTEXT_MAX_POINTS
    point_dim: int = HOUSE_POINT_DIM
    embed_dim: int = 1024
    input_key: str = HOUSE_CONTEXT_KEY

    @nn.compact
    def __call__(
        self,
        obs: Mapping[str, jax.Array],
        *,
        train: bool = False,
    ) -> jax.Array:
        """Encode R2Dreamer point-cloud observations into RSSM embeddings."""
        _ = (obs, train)
        raise NotImplementedError(
            "PointNet2FeatureEncoder is a skeleton. Implement "
            f"obs[{self.input_key!r}] -> (..., {self.embed_dim}) here."
        )

    def encode_points(
        self,
        points: jax.Array,
        *,
        train: bool = False,
    ) -> PointNet2BackboneOutput:
        """Adapt PointNet++ point tensors to backbone embeddings/endpoints."""
        _ = (points, train)
        raise NotImplementedError(
            "PointNet2FeatureEncoder.encode_points is a skeleton for the "
            f"{self.variant!r} PointNet++ backbone."
        )


class PointNet2Encoder(Encoder):
    """Launcher-side encoder-selection skeleton for PointNet++.

    ``Encoder.spec()``, ``make_adapter()``, and ``new_adapter()`` are inherited
    from the R2Dreamer launcher-side ``Encoder`` interface. Once implemented,
    the missing adapter should expose ``PointNet2PipelineSpec.point_shape`` as
    the encoder input and ``PointNet2FeatureEncoder`` as the module class.
    """

    encoder_type = "pointnet2"
    module_cls = PointNet2FeatureEncoder
    env_render_resolution = 64
    design_notes = (
        "Skeleton for adapting external TensorFlow PointNet++ point-cloud "
        "features into the R2Dreamer encoder pipeline."
    )

    def __init__(self, pipeline_spec: PointNet2PipelineSpec | None = None):
        self.pipeline_spec = pipeline_spec or PointNet2PipelineSpec()

    @classmethod
    def from_train_args(cls, args: Any) -> PointNet2Encoder:
        """Build the skeleton selection from parsed training arguments."""
        _ = (cls, args)
        raise NotImplementedError(
            "PointNet2Encoder.from_train_args is a skeleton. Add CLI/config "
            "fields before wiring this encoder into launch registries."
        )

    def _build_adapter(self):
        raise NotImplementedError(
            "PointNet2Encoder is a skeleton. Add PointNet2ObsAdapter before "
            "registering this encoder for training or evaluation."
        )
