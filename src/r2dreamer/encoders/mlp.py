"""MLP-based encoder modules for flat and hybrid observations."""

from collections.abc import Mapping

import flax.linen as nn
import jax.numpy as jnp
from flax import struct
from jax.typing import DTypeLike

from src.r2dreamer.encoders.cnn import ConvEncoder, _symlog, make_rgb_conv_encoder
from src.r2dreamer.encoders.constants import (
    AGG_RAW_DIM,
    HYBRID_RGB_DIM,
    HYBRID_VGGT_DIM,
)
from src.r2dreamer.encoders.shape_utils import flatten_event, restore_leading
from src.r2dreamer.observation_keys import (
    CAMERA_POSE_KEY,
    CAMERA_TOKEN_GLOBAL_KEY,
    GLOBAL_PATCH_TOKENS_KEY,
    HOUSE_CONTEXT_KEY,
    HOUSE_CONTEXT_SIZE_KEY,
    HYBRID_IMAGE_KEY,
    HYBRID_WP_CP_KEY,
    WORLD_POINTS_KEY,
)
from src.r2dreamer.world_model.heads import R2MLP
from src.r2dreamer.world_model.rssm import RMSNorm


class MLPEncoder(nn.Module):
    """Generic flat-feature MLP encoder.

    Parameters:
        embed_dim: Output embedding width consumed by the RSSM posterior.
        hidden: Width of each hidden block.
        num_layers: Number of hidden ``Dense -> RMSNorm -> SiLU`` blocks.
            ``num_layers=0`` is a linear encoder: only ``Dense(embed_dim)``.

    Returns:
        A float array with shape ``(..., embed_dim)``. Leading dimensions are
        preserved by Flax dense broadcasting.
    """

    embed_dim: int = 1024
    hidden: int = 1024
    num_layers: int = 1

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Encode flat or structured WP/CP observations into RSSM embeddings."""
        if isinstance(obs, Mapping):
            if "features" in obs:
                x = obs["features"]
            else:
                world_points = jnp.asarray(obs[WORLD_POINTS_KEY])
                camera_pose = jnp.asarray(obs[CAMERA_POSE_KEY])
                world_points = world_points.reshape(*world_points.shape[:-3], -1)
                camera_pose = camera_pose.reshape(*camera_pose.shape[:-1], -1)
                x = jnp.concatenate([world_points, camera_pose], axis=-1)
        else:
            x = obs
        x = jnp.asarray(x)
        for i in range(self.num_layers):
            x = nn.Dense(self.hidden, name=f"hidden{i}")(x)
            x = RMSNorm(name=f"norm{i}")(x)
            x = nn.silu(x)
        return nn.Dense(self.embed_dim, name="proj")(x)


class WP64CNNCPMLPEncoder(nn.Module):
    """CNN over 64x64 world points plus MLP over camera pose."""

    embed_dim: int = 1024
    conv_depth: int = 16
    conv_kernel: int = 5
    conv_mults: tuple[int, ...] = (2, 3, 4, 4)
    cp_hidden: int = 128
    cp_layers: int = 1

    @nn.compact
    def __call__(self, obs: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Encode structured ``world_points`` and ``camera_pose`` fields."""
        if not isinstance(obs, dict):
            raise TypeError("WP64CNNCPMLPEncoder expects structured obs")
        world_points = jnp.asarray(obs[WORLD_POINTS_KEY], dtype=jnp.float32)
        camera_pose = jnp.asarray(obs[CAMERA_POSE_KEY], dtype=jnp.float32)
        world_points_embed = ConvEncoder(
            depth=self.conv_depth,
            kernel_size=self.conv_kernel,
            mults=self.conv_mults,
            input_kind="world_points",
            name="wp_conv",
        )(world_points)
        camera_pose_embed = camera_pose
        for i in range(self.cp_layers):
            camera_pose_embed = nn.Dense(self.cp_hidden, name=f"cp_hidden{i}")(
                camera_pose_embed
            )
            camera_pose_embed = RMSNorm(name=f"cp_norm{i}")(camera_pose_embed)
            camera_pose_embed = nn.silu(camera_pose_embed)
        camera_pose_embed = nn.Dense(self.cp_hidden, name="cp_proj")(camera_pose_embed)
        fused = jnp.concatenate([world_points_embed, camera_pose_embed], axis=-1)
        return nn.Dense(self.embed_dim, name="proj")(fused)


class HousePointsCameraEncoder(nn.Module):
    """Encode current camera pose against one house point cloud.

    ``camera_pose`` is per replay step with shape ``(B*T, 9)``. ``house_points``
    is a singleton snapshot with shape ``(1, N, 6)`` or ``(N, 6)`` and is
    broadcast across all camera poses after the house branch is pooled. When
    ``HOUSE_CONTEXT_SIZE_KEY`` is present in the obs, rows past that count are
    treated as zero padding and masked out of the mean/max pooling.

    Attributes:
        house_point_norm: Normalization applied to the house-point metric XYZ
            channels ``[:3]`` before the point MLP. ``"symlog"`` (default)
            compresses the unbounded metric range with ``sign(x)*log1p(|x|)``
            and leaves the RGB channels ``[3:]`` untouched; ``"none"`` passes
            the raw coordinates through. Any other value raises ``ValueError``.
    """

    embed_dim: int = 1024
    camera_hidden: int = 1024
    camera_layers: int = 1
    point_hidden: int = 256
    point_layers: int = 2
    camera_pose_dim: int = 9
    house_point_dim: int = 6
    house_point_norm: str = "symlog"
    compute_dtype: DTypeLike = jnp.float32

    def _camera_embedding(self, camera_pose: jnp.ndarray) -> jnp.ndarray:
        if camera_pose.ndim < 1 or camera_pose.shape[-1] != self.camera_pose_dim:
            raise ValueError(
                f"camera_pose must have shape (..., {self.camera_pose_dim}), "
                f"got {camera_pose.shape}"
            )
        x = camera_pose.astype(self.compute_dtype)
        for i in range(self.camera_layers):
            x = nn.Dense(
                self.camera_hidden, name=f"camera_hidden{i}", dtype=self.compute_dtype
            )(x)
            x = RMSNorm(name=f"camera_norm{i}")(x)
            x = nn.silu(x)
        return nn.Dense(self.embed_dim, name="camera_proj", dtype=self.compute_dtype)(x)

    def _house_embedding(
        self,
        house_points: jnp.ndarray,
        batch_size: int,
        house_size: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        if house_points.ndim == 2:
            house_points = house_points[None]
        if house_points.ndim != 3 or house_points.shape[-1] != self.house_point_dim:
            raise ValueError(
                "house_points must have shape (N, 6) or (S, N, 6), "
                f"got {house_points.shape}"
            )
        x = house_points.astype(self.compute_dtype)
        # Normalize the metric XYZ channels [:3] before the point MLP; RGB
        # channels [3:] are already in [0, 1] and pass through untouched.
        if self.house_point_norm == "symlog":
            x = jnp.concatenate([_symlog(x[..., :3]), x[..., 3:]], axis=-1)
        elif self.house_point_norm != "none":
            raise ValueError(
                "house_point_norm must be 'symlog' or 'none', got "
                f"{self.house_point_norm!r}"
            )
        for i in range(self.point_layers):
            x = nn.Dense(
                self.point_hidden, name=f"point_hidden{i}", dtype=self.compute_dtype
            )(x)
            x = RMSNorm(name=f"point_norm{i}")(x)
            x = nn.silu(x)
        # Rows >= house_size are zero padding from the fixed-shape snapshot;
        # mask them so pooled statistics reflect only real points. The house
        # cloud is a singleton broadcast across the batch, so a single size
        # scalar governs every row; callers without the size key (legacy
        # paths) get an all-valid mask, i.e. plain mean/max pooling.
        n_points = house_points.shape[1]
        if house_size is None:
            house_size = n_points
        size = jnp.asarray(house_size, dtype=jnp.int32).reshape(-1)[:1]
        valid = (jnp.arange(n_points)[None, :] < size[:, None])[..., None]
        denom = jnp.maximum(size, 1).astype(x.dtype)[:, None]
        mean = (x * valid).sum(axis=1) / denom
        maxp = jnp.where(valid, x, -jnp.inf).max(axis=1)
        maxp = jnp.where((size > 0)[:, None], maxp, jnp.zeros_like(maxp))
        pooled = jnp.concatenate([mean, maxp], axis=-1)
        house_embed = nn.Dense(
            self.embed_dim, name="house_proj", dtype=self.compute_dtype
        )(pooled)
        if house_embed.shape[0] == 1 and batch_size != 1:
            house_embed = jnp.broadcast_to(house_embed, (batch_size, self.embed_dim))
        elif house_embed.shape[0] != batch_size:
            raise ValueError(
                "house_points batch must be singleton or match camera batch: "
                f"house={house_embed.shape[0]}, camera={batch_size}"
            )
        return house_embed

    def _branches(self, obs: dict[str, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        if not isinstance(obs, dict):
            raise TypeError("HousePointsCameraEncoder expects structured obs")
        camera_pose, leading_shape = flatten_event(obs[CAMERA_POSE_KEY], event_ndims=1)
        camera_embed = self._camera_embedding(camera_pose)
        house_size = obs.get(HOUSE_CONTEXT_SIZE_KEY)
        house_embed = self._house_embedding(
            jnp.asarray(obs[HOUSE_CONTEXT_KEY]),
            batch_size=camera_pose.shape[0],
            house_size=house_size,
        )
        return restore_leading(camera_embed, leading_shape), restore_leading(
            house_embed, leading_shape
        )

    @nn.compact
    def __call__(self, obs: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Return fused ``[camera_pose_embedding | house_points_embedding]``."""
        camera_embed, house_embed = self._branches(obs)
        return jnp.concatenate([camera_embed, house_embed], axis=-1)

    @nn.compact
    def branches(self, obs: dict[str, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Diagnostic split: ``(camera_pose_embedding, house_points_embedding)``."""
        return self._branches(obs)


class HybridHousePointsCameraEncoder(HousePointsCameraEncoder):
    """Additive hybrid: RGB CNN backbone plus gated pose/house-points branches.

    Extends ``HousePointsCameraEncoder`` with the image-only CNN baseline as
    an ungated backbone branch and puts zero-initialized scalar gates (the
    ``HybridEncoder`` pattern) on the camera-pose and pooled house-points
    branches. At initialization the embedding therefore equals
    ``[cnn_embed | 0 | 0]`` — exactly the information the CNN baseline sees —
    and training must learn to admit the extra 3D context, so any gain over
    the baseline is attributable to the map/pose rather than to architecture
    drift (docs/notes/2026-07-06-house-points-vs-cnn-baseline-review.md §5).

    Consumes ``HYBRID_IMAGE_KEY`` (rgb64) in addition to the parent's
    ``camera_pose`` / ``house_context`` (+ size) keys; the fused embedding is
    ``[cnn | gate_camera * camera | gate_house * house]`` of width
    ``1024 + 2 * embed_dim``.
    """

    cnn_depth: int = 16
    cnn_kernel: int = 5
    cnn_mults: tuple[int, ...] = (2, 3, 4, 4)

    def _hybrid_branches(
        self, obs: dict[str, jnp.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return ``(cnn, gated_camera, gated_house, gate_camera, gate_house)``."""
        if not isinstance(obs, Mapping):
            raise TypeError("HybridHousePointsCameraEncoder expects structured obs")
        cnn_embed = make_rgb_conv_encoder(
            depth=self.cnn_depth,
            kernel_size=self.cnn_kernel,
            mults=self.cnn_mults,
            name="cnn",
            compute_dtype=self.compute_dtype,
        )(jnp.asarray(obs[HYBRID_IMAGE_KEY]))
        camera_embed, house_embed = self._branches(obs)
        gate_camera = self.param("gate_camera", nn.initializers.zeros, ())
        gate_house = self.param("gate_house", nn.initializers.zeros, ())
        # Cast the float32 scalar gates down so the gated products (and the
        # fused concat) keep the branches' compute dtype instead of promoting
        # the whole embedding back to float32.
        return (
            cnn_embed,
            gate_camera.astype(camera_embed.dtype) * camera_embed,
            gate_house.astype(house_embed.dtype) * house_embed,
            gate_camera,
            gate_house,
        )

    @nn.compact
    def __call__(self, obs: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Return fused ``[cnn | gated camera_pose | gated house_points]``."""
        cnn_embed, camera_embed, house_embed, _, _ = self._hybrid_branches(obs)
        return jnp.concatenate([cnn_embed, camera_embed, house_embed], axis=-1)

    @nn.compact
    def branches(
        self, obs: dict[str, jnp.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Diagnostic split: ``(cnn, gated_camera, gated_house, gate_camera, gate_house)``."""
        return self._hybrid_branches(obs)


@struct.dataclass
class HouseGlobalObs:
    """Structured obs for the house-global-embedding encoder (a JAX pytree).

    Attributes:
      global_patch_tokens: ``(…, num_patch_tokens, token_dim)`` VGGT global patch
        tokens — always present.
      image: ``(…, 3, 64, 64)`` per-step RGB frame; present in hybrid mode.
      camera_token_global: ``(…, 1, token_dim)`` camera token; present in camera
        mode.
    """

    global_patch_tokens: jnp.ndarray
    image: jnp.ndarray | None = None
    camera_token_global: jnp.ndarray | None = None


class TokenReducer(nn.Module):
    """PointNet-style token reducer.

    Patch tokens ``(…, N, token_dim)`` are processed by a shared per-token MLP
    then max-pooled over the token axis. Singleton tokens ``(…, 1, token_dim)``
    skip the pool and are squeezed to ``(…, token_dim)`` first. Both paths use
    ``Dense -> RMSNorm -> silu`` followed by a projection to ``embed_dim``.

    Attributes:
      embed_dim: Output width.
      hidden: MLP hidden width.
      layers: Number of MLP blocks before projection.
      pool_tokens: When True (default), max-pool over the token axis.
      token_dim: Expected singleton-token width (validated when pool_tokens=False).
    """

    output_dim: int = 1024
    hidden: int = 1024
    layers: int = 1
    pool_tokens: bool = True

    @nn.compact
    def __call__(self, tokens: jnp.ndarray) -> jnp.ndarray:
        """Reduce patch or singleton tokens to ``(…, output_dim)``."""
        x = tokens
        for i in range(self.layers):
            x = nn.Dense(self.hidden, name=f"hidden{i}")(x)
            x = RMSNorm(name=f"norm{i}")(x)
            x = nn.silu(x)
        if self.pool_tokens:
            x = x.max(axis=-2)  # PointNet's symmetric g-pool over the token axis
        else:
            x = x[..., 0, :]  # squeeze the singleton token axis
        return nn.Dense(self.output_dim, name="proj")(x)


class HouseGlobalEmbeddingEncoder(nn.Module):
    """Flax module wrapper around the house-global-embedding encoder logic.

    This class keeps the encoder callable through the standard Flax
    ``init``/``apply`` API while exposing the implementation under the
    requested snake_case name. The functional logic is delegated to
    :func:`_encode_house_global_obs_impl`.

    Attributes:
        mlp_layers: Number of hidden ``Dense -> RMSNorm -> silu`` blocks in
            each TokenReducer branch.
        hidden_dim: Hidden width of each TokenReducer branch.
    """

    mlp_layers: int = 1
    hidden_dim: int = 1024

    @nn.compact
    def __call__(self, obs: HouseGlobalObs | Mapping[str, jnp.ndarray] ) -> jnp.ndarray:
        """Encode a :class:`HouseGlobalObs` into an RSSM embedding."""
        if isinstance(obs, Mapping):
              obs = HouseGlobalObs(
                  global_patch_tokens=obs[GLOBAL_PATCH_TOKENS_KEY],
                  image=obs.get(HYBRID_IMAGE_KEY),
                  camera_token_global=obs.get(CAMERA_TOKEN_GLOBAL_KEY),
              )
        house_embed = TokenReducer(
            hidden=self.hidden_dim,
            layers=self.mlp_layers,
            name="house",
        )(obs.global_patch_tokens)
        if obs.image is not None:
            first_embed = ConvEncoder(name="rgb")(obs.image)
            return jnp.concatenate([first_embed, house_embed], axis=-1)
        if obs.camera_token_global is not None:
            first_embed = TokenReducer(
                hidden=self.hidden_dim,
                layers=self.mlp_layers,
                pool_tokens=False,
                name="camera",
            )(obs.camera_token_global)
            return jnp.concatenate([first_embed, house_embed], axis=-1)
        return house_embed


class VGGTAggregatorMLPEncoder(nn.Module):
    """MLP encoder for pre-pooled aggregator features.

    Input layout is ``[cam | mean_patches | max_patches]`` with shape
    ``(B, 3 * pool_dim)``. Each slice is normalized separately before the hidden
    MLP blocks so scale differences between readouts do not bleed into fusion.
    """

    embed_dim: int = 1024
    pool_dim: int = 1024
    hidden: int = 1024
    num_layers: int = 1

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Encode pooled token features into an RSSM embedding."""
        if isinstance(obs, Mapping):
            obs = obs["features"]
        obs = jnp.asarray(obs)
        if obs.ndim < 1 or obs.shape[-1] != 3 * self.pool_dim:
            raise ValueError(
                f"expected (..., {3 * self.pool_dim}) pooled features, got {obs.shape}"
            )
        camera, mean_patches, max_patches = jnp.split(obs, 3, axis=-1)
        camera = RMSNorm(name="norm_cam")(camera)
        mean_patches = RMSNorm(name="norm_mean")(mean_patches)
        max_patches = RMSNorm(name="norm_max")(max_patches)
        x = jnp.concatenate([camera, mean_patches, max_patches], axis=-1)
        for i in range(self.num_layers):
            x = nn.Dense(self.hidden, name=f"hidden{i}")(x)
            x = RMSNorm(name=f"norm{i}")(x)
            x = nn.silu(x)
        return nn.Dense(self.embed_dim, name="proj")(x)


class VGGTAggRawMLPEncoder(nn.Module):
    """MLP over flattened raw aggregator tokens.

    Input ``(B, AGG_RAW_DIM)`` is camera token plus patch tokens with register
    tokens dropped. Replay stores float16; the cast to float32 here lets dense
    matmuls run in the agent's working precision.
    """

    embed_dim: int = 1024
    hidden: int = 1024
    num_layers: int = 3

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Encode flattened raw token features into an RSSM embedding."""
        if isinstance(obs, Mapping):
            obs = obs["features"]
        obs = jnp.asarray(obs)
        if obs.ndim < 1 or obs.shape[-1] != AGG_RAW_DIM:
            raise ValueError(
                f"VGGTAggRawMLPEncoder expects (..., {AGG_RAW_DIM}), got {obs.shape}"
            )
        x = obs.astype(jnp.float32)
        for i in range(self.num_layers):
            x = nn.Dense(self.hidden, name=f"hidden{i}")(x)
            x = RMSNorm(name=f"norm{i}")(x)
            x = nn.silu(x)
        return nn.Dense(self.embed_dim, name="proj")(x)


class HybridEncoder(nn.Module):
    """Hybrid CNN + gated MLP encoder feeding both modalities to the latent.

    Input may be a structured ``{"image": ..., "wp_cp": ...}`` observation or
    the legacy packed vector ``[rgb64_flat | flat_features]``. Leading dimensions
    such as replay ``(B, T)`` are preserved.
    """

    cnn_depth: int = 16
    cnn_kernel: int = 5
    cnn_mults: tuple[int, ...] = (2, 3, 4, 4)
    vggt_embed_dim: int = 1024
    mlp_hidden: int = 1024
    mlp_layers: int = 2
    rgb_dim: int = HYBRID_RGB_DIM
    vggt_dim: int = HYBRID_VGGT_DIM

    def _branches(
        self, obs: jnp.ndarray | Mapping[str, jnp.ndarray]
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return ``(cnn_embed, gated_feature_embed, gate)`` from hybrid obs."""
        if isinstance(obs, Mapping):
            rgb = jnp.asarray(obs[HYBRID_IMAGE_KEY])
            context_key = (
                HYBRID_WP_CP_KEY if HYBRID_WP_CP_KEY in obs else HOUSE_CONTEXT_KEY
            )
            features = jnp.asarray(obs[context_key])
            if features.shape[-1] != self.vggt_dim:
                raise ValueError(
                    f"expected hybrid context width {self.vggt_dim}, got {features.shape}"
                )
        else:
            obs = jnp.asarray(obs)
            if obs.ndim < 1 or obs.shape[-1] != self.rgb_dim + self.vggt_dim:
                raise ValueError(
                    f"expected (..., {self.rgb_dim + self.vggt_dim}) hybrid features, "
                    f"got {obs.shape}"
                )
            rgb = obs[..., : self.rgb_dim].reshape(*obs.shape[:-1], 3, 64, 64)
            features = obs[..., self.rgb_dim :]
        cnn_embed = make_rgb_conv_encoder(
            depth=self.cnn_depth,
            kernel_size=self.cnn_kernel,
            mults=self.cnn_mults,
            name="cnn",
        )(rgb)
        feature_mlp = R2MLP(
            hidden=self.mlp_hidden,
            layers=self.mlp_layers,
            out_dim=self.vggt_embed_dim,
            name="vggt_mlp",
        )
        gate = self.param("gate", nn.initializers.zeros, ())
        feature_embed = gate * feature_mlp(features)
        return cnn_embed, feature_embed, gate

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Encode packed hybrid observations into one concatenated embedding."""
        cnn_embed, feature_embed, _ = self._branches(obs)
        return jnp.concatenate([cnn_embed, feature_embed], axis=-1)

    @nn.compact
    def branches(
        self, obs: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Diagnostic split: ``(cnn_embed, gated_feature_embed, gate_scalar)``."""
        return self._branches(obs)
