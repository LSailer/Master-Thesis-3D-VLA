"""PointNet cloud branch and the T-Net alignment network it is built from.

Follows Qi et al. (arXiv:1612.00593): input/feature T-Nets, shared per-point
MLPs, max pool. ``PointNetCloudEncoder`` is the branch the routed composite
encoder instantiates for ``Encoder.POINTNET`` fields.

Adaptations to this codebase:
    - Norms are RMSNorm + SiLU rather than the paper's BatchNorm + ReLU: the
      encoder call has no train flag or mutable batch-stats plumbing, and
      RMSNorm + SiLU is the repo idiom.
    - The output layer is zero-kernel/identity-bias initialized, so the
      predicted transform starts as the identity.
    - Dense/matmul compute runs in ``compute_dtype`` (default bfloat16, the
      repo default). Parameters stay float32 (Flax ``param_dtype`` default).
    - The paper's feature-transform orthogonality regularizer is not wired
      into the training losses.
"""

from __future__ import annotations

import flax.linen as nn
import jax.numpy as jnp
from jax.typing import DTypeLike

from src.r2dreamer.encoders.mlp import RMSNorm


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


class PointNetCloudEncoder(nn.Module):
    """Classic PointNet over one unbatched ``(N, point_dim)`` cloud.

    Same architecture as ``PointNetHousePointsCameraEncoder._house_embedding``
    (input/feature T-Nets, shared MLPs, max pool, optional projection) without
    the snapshot/camera plumbing: the input is the raw live cloud, subsampled
    with an even stride to at most ``num_points`` rows.

    TODO(house-padding): a cloud whose N changes between steps recompiles the
    branch on every new N. Adapters therefore emit a fixed-size cloud; see
    docs/notes/adapter-routing-migration.md for the follow-up.
    """

    num_points: int = 16384
    point_dim: int = 6
    embed_dim: int = 1024
    tnet_mlp: tuple[int, ...] = (64, 128, 1024)
    tnet_fc: tuple[int, ...] = (512, 256)
    mlp1: tuple[int, ...] = (64, 64)
    mlp2: tuple[int, ...] = (64, 128, 1024)
    compute_dtype: DTypeLike = jnp.bfloat16

    @nn.compact
    def __call__(self, points: jnp.ndarray) -> jnp.ndarray:
        """Encode ``(N, point_dim)`` points into one ``(embed_dim,)`` vector."""
        if points.ndim != 2 or points.shape[-1] != self.point_dim:
            raise ValueError(
                f"expected (N, {self.point_dim}) cloud, got {points.shape}"
            )
        n_points = points.shape[0]
        m = min(self.num_points, n_points)
        sample_idx = (jnp.arange(m, dtype=jnp.int32) * n_points) // m
        # Centering/scale in float32: metric world offsets exceed reduced-
        # precision resolution, and the 1e-6 floor below is subnormal in
        # float16 - it flushes to zero, turning a degenerate (zero-extent)
        # cloud into 0/0 = NaN. The normalized cloud is O(1), so casting down
        # afterwards is safe.
        xyz = jnp.asarray(points[sample_idx, :3], jnp.float32)
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
            x = nn.Dense(width, dtype=self.compute_dtype, name=f"mlp1_{i}")(x)
            x = RMSNorm(name=f"mlp1_norm{i}")(x)
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
            x = nn.Dense(width, dtype=self.compute_dtype, name=f"mlp2_{i}")(x)
            x = RMSNorm(name=f"mlp2_norm{i}")(x)
            x = nn.silu(x)

        embed = x.max(axis=0)
        if self.mlp2[-1] != self.embed_dim:
            embed = nn.Dense(
                self.embed_dim, dtype=self.compute_dtype, name="proj"
            )(embed)
        return embed
