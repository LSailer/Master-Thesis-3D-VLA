"""VGGTObsAdapter: wraps VGGTFeatureExtractor for RL acting."""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import numpy as np

from src.r2dreamer.adapters.obs_adapter import ObsAdapter
from src.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor as VGGTFeatureExtractor


VGGT_FEATURE_DIM = 4116  # 37*37*3 + 9
VGGTFeatureKind = Literal["wp_cp", "aggregator", "wp_dense", "agg_raw"]

# Raw aggregator readout: drop the 4 register tokens, keep camera(1) + patches.
# At the default 518² / 37x37-patch / 1024-d config: 1 + 1369 = 1370 tokens.
AGG_RAW_TOKENS = 1370            # cam(1) + patches(1369); registers (idx 1:5) dropped
AGG_RAW_DIM = AGG_RAW_TOKENS * 1024  # 1,402,880 at the default 1024-d embedding


def flatten_world_points_camera_pose(out: dict) -> jnp.ndarray:
    """Flatten VGGT outputs into a single feature vector (JAX)."""
    wp = out["world_points"].reshape(-1)  # (4107,)
    cp = out["camera_pose"]              # (9,)
    return jnp.concatenate([wp, cp]).astype(jnp.float32)


def dense_world_points_chw(out: dict) -> jnp.ndarray:
    """Return the full-resolution world-point map in (3, H, W) layout (3D-53).

    ``out["dense_world_points"]`` is the pre-pool DPT point map at (H, W, 3) —
    one metric XYZ point per pixel. We transpose to channel-first so it matches
    the ``(B, C, H, W)`` layout the conv encoder expects (XYZ as a 3-channel
    image). No camera pose is appended: a 9-vector cannot be a spatial channel,
    so this variant is intentionally WP-only (a WP+CP hybrid is a follow-up).
    """
    dense = out["dense_world_points"]          # (H, W, 3)
    return jnp.transpose(dense, (2, 0, 1)).astype(jnp.float32)  # (3, H, W)


_PATCH_START_IDX = 5  # 1 camera token + 4 register tokens, then patches


def pool_aggregator_tokens(out: dict, expected_shape: tuple[int, ...]) -> jnp.ndarray:
    """Return three pre-head pools concatenated as a single (3*D,) vector.

    Layout matches the aggregator's ``[camera, register, patches]`` ordering
    (see ``src/vggt/jax/aggregator.py``): the camera token at index 0 is
    the embedding VGGT's own ``camera_head`` reads to predict pose, so we keep
    it unmixed. Register tokens (1:5) are attention sinks and dropped. Patches
    (5:) are reduced with both mean (smooth global) and max (salient features)
    so the encoder sees signals at different scales.
    """
    features = out["aggregator_features"]
    if features.shape != expected_shape:
        raise ValueError(
            f"expected aggregator_features shape {expected_shape}, "
            f"got {features.shape}"
        )
    features = features.astype(jnp.float32)
    cam = features[0]
    patches = features[_PATCH_START_IDX:]
    mean_p = patches.mean(axis=0)
    max_p = patches.max(axis=0)
    return jnp.concatenate([cam, mean_p, max_p], axis=0)


def flatten_raw_aggregator(out: dict, expected_shape: tuple[int, ...]) -> jnp.ndarray:
    """Flatten the RAW aggregator tokens, dropping the 4 register tokens (JAX).

    Unlike ``pool_aggregator_tokens`` (which collapses patches to mean+max), this
    keeps every token: camera token (idx 0) + all patch tokens (idx
    ``_PATCH_START_IDX:``) = ``1 + (P-5)`` tokens, flattened row-major to
    ``(n_tokens * embed_dim,)`` — 1370*1024 = 1,402,880 at the default config. The
    4 register tokens (idx 1:5) are attention sinks and dropped, matching the
    pooled path. Stored float16 in replay (~2.81 MB/frame).
    """
    features = out["aggregator_features"]
    if features.shape != expected_shape:
        raise ValueError(
            f"expected aggregator_features shape {expected_shape}, got {features.shape}"
        )
    features = features.astype(jnp.float32)
    cam = features[0:1]                          # (1, D)
    patches = features[_PATCH_START_IDX:]        # (P-5, D)
    kept = jnp.concatenate([cam, patches], axis=0)  # (1 + P-5, D)
    return kept.reshape(-1)                       # (n_tokens * D,)


class VGGTObsAdapter(ObsAdapter):
    """Runs VGGT extraction, returns features for both buffer and agent."""

    def __init__(self, extractor: VGGTFeatureExtractor, feature_kind: VGGTFeatureKind = "wp_cp"):
        if feature_kind == "wp_cp":
            # WP grid is configurable (37 default, 64 for the hi-res variant);
            # flat dim = K*K*3 world points + 9 camera-pose. Defaults to 4116.
            k = int(getattr(extractor, "wp_pool_size", 37))
            buffer_shape = (k * k * 3 + 9,)
            buffer_dtype = "float32"
        elif feature_kind == "aggregator":
            embed_dim = int(extractor.aggregator_feature_shape[-1])
            buffer_shape = (3 * embed_dim,)  # [cam | mean_patches | max_patches]
            buffer_dtype = "float32"
        elif feature_kind == "wp_dense":
            # Full-res world-point map as a (3, H, W) image (3D-53). Stored as
            # float16 to halve the per-frame cost (~3.2 MB -> ~1.6 MB at 518²).
            img = int(extractor.image_size)
            buffer_shape = (3, img, img)
            buffer_dtype = "float16"
        elif feature_kind == "agg_raw":
            # Raw flattened aggregator: drop the 4 register tokens, keep
            # cam(1) + patches -> (1 + (P - _PATCH_START_IDX)) * embed_dim.
            # ~1.40M float16 = ~2.81 MB/frame at defaults; keep the buffer small.
            shp = extractor.aggregator_feature_shape
            n_tokens = 1 + (int(shp[0]) - _PATCH_START_IDX)
            embed_dim = int(shp[-1])
            buffer_shape = (n_tokens * embed_dim,)
            buffer_dtype = "float16"
        else:
            raise ValueError(f"unknown VGGT feature_kind {feature_kind!r}")
        super().__init__(
            buffer_dtype=buffer_dtype,
            buffer_shape=buffer_shape,
            normalize_on_sample=False,
            on_episode_reset=extractor.reset,
        )
        self._extractor = extractor
        self._feature_kind: VGGTFeatureKind = feature_kind
        self._aggregator_feature_shape = tuple(getattr(extractor, "aggregator_feature_shape", ()))

    def transform(self, obs_dict: dict) -> tuple[np.ndarray, dict]:
        # wp_dense needs the pre-pool 518² point map; the other kinds don't, so
        # only request (and pay for materializing) the dense map when consumed.
        if self._feature_kind == "wp_dense":
            out = self._extractor.extract(obs_dict["image"], return_dense=True)
        else:
            out = self._extractor.extract(obs_dict["image"])
        if self._feature_kind == "aggregator":
            features_jax = pool_aggregator_tokens(out, self._aggregator_feature_shape)
        elif self._feature_kind == "agg_raw":
            features_jax = flatten_raw_aggregator(out, self._aggregator_feature_shape)
        elif self._feature_kind == "wp_dense":
            features_jax = dense_world_points_chw(out)
            if tuple(features_jax.shape) != tuple(self.buffer_shape):
                # Guards against a render_resolution / extractor-image-size mismatch:
                # fail loud here rather than via a cryptic broadcast error in the buffer.
                raise ValueError(
                    f"dense world-point map {tuple(features_jax.shape)} != declared "
                    f"buffer shape {tuple(self.buffer_shape)}; the env render resolution "
                    f"must equal the VGGT extractor image size ({self.buffer_shape[-1]})."
                )
        else:
            features_jax = flatten_world_points_camera_pose(out)

        # The replay buffer is CPU/NumPy storage. The acting path keeps JAX
        # float32 features so it can feed the JIT-compiled agent directly.
        replay_features = np.asarray(features_jax)
        if self._feature_kind == "aggregator":
            replay_features = replay_features.astype(np.float32)
        elif self._feature_kind == "agg_raw":
            # Match the float16 buffer storage declared in __init__.
            replay_features = replay_features.astype(np.float16)
        elif self._feature_kind == "wp_dense":
            # Match the float16 buffer storage declared in __init__.
            replay_features = replay_features.astype(np.float16)

        agent_features = features_jax.astype(jnp.float32)
        agent_obs = {"features": agent_features, "is_first": obs_dict.get("is_first", False)}
        return replay_features, agent_obs
