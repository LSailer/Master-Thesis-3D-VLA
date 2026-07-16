"""Classic PointNet house-context encoder for the live house-points-pose pipeline.

Replaces the house branch of ``HousePointsCameraEncoder`` (per-point MLP +
one global masked pool) with the vanilla PointNet feature backbone
(Qi et al., arXiv:1612.00593):

    xyz (n, 3)
      -> input T-Net (3x3 transform)
      -> shared MLP (64, 64)             per point: 3 -> 64 -> 64
      -> feature T-Net (64x64 transform)
      -> shared MLP (64, 128, 1024)      per point: 64 -> 64 -> 128 -> 1024
      -> max pool over the n points      -> (1, 1024) global feature

Adaptations to this codebase:
    - Norms are RMSNorm + SiLU rather than the paper's BatchNorm + ReLU: the
      encoder call has no train flag or mutable batch-stats plumbing, and
      RMSNorm + SiLU is the repo idiom (same substitution as gnn_house.py).
    - The fixed-shape snapshot is even-stride subsampled over the valid prefix
      to ``num_points`` rows (same scheme as the GNN encoder), so every
      selected point is real and no pool needs a padding mask. Rows repeat
      when the map is still smaller than ``num_points``, which is harmless
      under max pooling.
    - xyz is centered and scale-normalized before the input T-Net (house
      clouds live in metric world coordinates with arbitrary offsets); the
      rgb columns of the 6-dim house points are dropped — classic PointNet
      consumes xyz only.
    - Both T-Net output layers are zero-kernel/identity-bias initialized, so
      each transform starts as the identity.
    - Dense/matmul compute runs in ``compute_dtype`` (default bfloat16, the
      repo default). Parameters stay float32 (Flax ``param_dtype`` default),
      and the point centering/scale statistics are computed in float32 before
      casting down — house clouds live in metric world coordinates whose
      offsets exceed bfloat16 precision.
    - The paper's feature-transform orthogonality regularizer is not wired
      into the training losses.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp
from jax.typing import DTypeLike

from src.r2dreamer.encoders.house_points_pose import VGGTHousePointsPoseEncoder
from src.r2dreamer.encoders.mlp import HousePointsCameraEncoder, RMSNorm
from src.r2dreamer.encoders.shape_utils import (
    singleton_house_cloud,
    validate_house_points,
)


def _identity_bias(k: int):
    """Build a bias initializer that emits a flattened (k, k) identity.

    Args:
      k: Side length of the square transform matrix.

    Returns:
      A Flax-compatible initializer ``(key, shape, dtype) -> jnp.ndarray``
      returning ``eye(k)`` reshaped to ``shape``.
    """

    def init(key, shape, dtype=jnp.float32):
        del key
        return jnp.eye(k, dtype=dtype).reshape(shape)

    return init


class TNet(nn.Module):
    """PointNet alignment network predicting one (k, k) transform matrix.

    Shared per-point MLP, max pool over points, then fully connected layers
    into a ``k * k`` output. The output Dense is zero-kernel/identity-bias
    initialized, so the predicted transform is exactly the identity at init
    (which also means the internal layers receive zero gradient on the very
    first step — the transform kernel itself does not, so training unblocks
    them after one update).

    Attributes:
      k: Side length of the square transform (3 for the input T-Net, 64 for
        the feature T-Net).
      mlp_widths: Per-point shared-MLP widths before the max pool.
      fc_widths: Fully connected widths after the max pool.
      compute_dtype: JAX/Flax compute dtype for the Dense layers and the
        predicted transform.
    """

    k: int
    mlp_widths: tuple[int, ...] = (64, 128, 1024)
    fc_widths: tuple[int, ...] = (512, 256)
    compute_dtype: DTypeLike = jnp.bfloat16

    @nn.compact
    def __call__(self, points: jnp.ndarray) -> jnp.ndarray:
        """Predict the alignment transform for one point set.

        Args:
          points: Point features with shape ``(n, k)``; cast to
            ``compute_dtype`` internally.

        Returns:
          The transform matrix with shape ``(k, k)``, ``compute_dtype``.
        """
        x = points.astype(self.compute_dtype)
        for i, width in enumerate(self.mlp_widths):
            x = nn.Dense(width, dtype=self.compute_dtype, name=f"mlp{i}")(x)
            x = RMSNorm(name=f"norm{i}")(x)
            x = nn.silu(x)
        pooled = x.max(axis=0)
        for i, width in enumerate(self.fc_widths):
            pooled = nn.Dense(width, dtype=self.compute_dtype, name=f"fc{i}")(pooled)
            pooled = RMSNorm(name=f"fc_norm{i}")(pooled)
            pooled = nn.silu(pooled)
        transform = nn.Dense(
            self.k * self.k,
            dtype=self.compute_dtype,
            kernel_init=nn.initializers.zeros,
            bias_init=_identity_bias(self.k),
            name="transform",
        )(pooled)
        return transform.reshape(self.k, self.k)


class PointNetHousePointsCameraEncoder(HousePointsCameraEncoder):
    """House branch = classic PointNet over a strided subset of the snapshot.

    Inherits the camera branch, obs plumbing, and singleton-broadcast
    behavior from ``HousePointsCameraEncoder``; only ``_house_embedding``
    changes. The max-pooled ``mlp2[-1]``-wide global feature is used as the
    house embedding directly when it matches ``embed_dim`` (the default:
    1024) and linearly projected otherwise.

    Attributes:
      num_points: Static point count fed to PointNet, taken as an even-stride
        subsample of the snapshot's valid prefix.
      tnet_mlp: Shared-MLP widths inside both T-Nets.
      tnet_fc: Fully connected widths inside both T-Nets.
      mlp1: Shared-MLP widths between the input and feature T-Nets; the last
        width sets the feature T-Net size.
      mlp2: Shared-MLP widths after the feature T-Net; the last width is the
        global-feature size.
      compute_dtype: JAX/Flax compute dtype for the PointNet Dense layers and
        transform matmuls; centering/scale statistics stay float32.
    """

    num_points: int = 16384
    tnet_mlp: tuple[int, ...] = (64, 128, 1024)
    tnet_fc: tuple[int, ...] = (512, 256)
    mlp1: tuple[int, ...] = (64, 64)
    mlp2: tuple[int, ...] = (64, 128, 1024)
    compute_dtype: DTypeLike = jnp.bfloat16

    def _house_embedding(
        self,
        house_points: jnp.ndarray,
        batch_size: int,
        house_size: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        if self.num_points < 1:
            raise ValueError(f"num_points must be >= 1, got {self.num_points}")
        house_points = validate_house_points(house_points, self.house_point_dim)
        points, n_points, size = singleton_house_cloud(
            house_points, house_size, "PointNet"
        )

        # Even stride over the valid prefix (same scheme as the GNN encoder);
        # rows repeat when size < m, which is harmless under max pooling.
        m = min(self.num_points, n_points)
        clamped = jnp.maximum(size, 1)
        sample_idx = (jnp.arange(m, dtype=jnp.int32) * clamped) // m
        xyz = points[sample_idx, :3]

        # Centering/scale in float32: metric world offsets exceed bfloat16
        # precision. The normalized cloud is O(1), so casting down is safe.
        center = xyz.mean(axis=0)
        scale = jnp.maximum(xyz.std(), 1e-6)
        xyz_n = ((xyz - center) / scale).astype(self.compute_dtype)

        input_transform = TNet(
            k=3,
            mlp_widths=self.tnet_mlp,
            fc_widths=self.tnet_fc,
            compute_dtype=self.compute_dtype,
            name="input_tnet",
        )(xyz_n)
        x = xyz_n @ input_transform
        for i, width in enumerate(self.mlp1):
            x = nn.Dense(width, dtype=self.compute_dtype, name=f"pointnet_mlp1_{i}")(x)
            x = RMSNorm(name=f"pointnet_mlp1_norm{i}")(x)
            x = nn.silu(x)
        feature_transform = TNet(
            k=self.mlp1[-1],
            mlp_widths=self.tnet_mlp,
            fc_widths=self.tnet_fc,
            compute_dtype=self.compute_dtype,
            name="feature_tnet",
        )(x)
        x = x.astype(self.compute_dtype) @ feature_transform
        for i, width in enumerate(self.mlp2):
            x = nn.Dense(width, dtype=self.compute_dtype, name=f"pointnet_mlp2_{i}")(x)
            x = RMSNorm(name=f"pointnet_mlp2_norm{i}")(x)
            x = nn.silu(x)

        house_embed = x.max(axis=0)[None]
        if self.mlp2[-1] != self.embed_dim:
            house_embed = nn.Dense(
                self.embed_dim, dtype=self.compute_dtype, name="pointnet_house_proj"
            )(house_embed)
        house_embed = jnp.where(size > 0, house_embed, jnp.zeros_like(house_embed))
        if batch_size != 1:
            house_embed = jnp.broadcast_to(house_embed, (batch_size, self.embed_dim))
        return house_embed


class PointNetHousePointsPoseEncoder(VGGTHousePointsPoseEncoder):
    """Launcher-side selection for the PointNet house encoder.

    Reuses the whole VGGT house-points-pose pipeline (adapter, live buffer,
    camera pose replay). The parent selection now defaults to the same
    PointNet module; this named selection pins it explicitly under the
    ``pointnet`` encoder type.
    """

    @property
    def encoder_type(self) -> str:
        return "pointnet"

    @property
    def module_cls(self) -> type[nn.Module]:
        return PointNetHousePointsCameraEncoder

    @property
    def design_notes(self) -> str:
        return (
            "Classic PointNet house branch — input/feature T-Nets, shared "
            "MLPs (64,64) and (64,128,1024), max pool to one 1024-d global "
            "feature — over an even-stride subset of the live house snapshot "
            "(src/r2dreamer/encoders/pointnet.py)."
        )
