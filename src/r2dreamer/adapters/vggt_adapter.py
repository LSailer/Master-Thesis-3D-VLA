"""VGGTObsAdapter: wraps VGGTFeatureExtractor for RL acting."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from src.r2dreamer.adapters.obs_adapter import ObsAdapter
from src.r2dreamer.obs_batch import CAMERA_POSE_KEY, WORLD_POINTS_KEY
from src.r2dreamer.observation_preparation.vggt import (
    VGGT_AGGREGATOR_PATCH_START_IDX,
    VGGT_AGGREGATOR_TOKEN_COUNT,
    VGGT_DEFAULT_AGGREGATOR_SHAPE,
    VGGT_FULL_TOKEN_EMBED_DIM,
    VGGTFeatureKind,
    aggregator_raw_dim,
    aggregator_token_dim,
    build_vggt_contract,
    contract_world_points_hwc_shape,
    head_readout_spec,
    wp_cp_dim,
)
from src.vggt.jax.feature_extractor import JAXVGGTFeatureExtractor as VGGTFeatureExtractor


VGGT_FEATURE_DIM = wp_cp_dim()  # 37*37*3 + 9

# Backwards-compatible aliases. Dimensions are derived from the shared VGGT
# observation-preparation constants instead of being hand-typed here.
AGG_RAW_DIM = aggregator_raw_dim(VGGT_DEFAULT_AGGREGATOR_SHAPE)
AGG_RAW_TOKENS = AGG_RAW_DIM // VGGT_DEFAULT_AGGREGATOR_SHAPE[-1]
AGG_TOKEN_TOKENS = VGGT_AGGREGATOR_TOKEN_COUNT
AGG_TOKEN_DIM = aggregator_token_dim(VGGT_DEFAULT_AGGREGATOR_SHAPE)
FULL_TOKEN_DIM = AGG_TOKEN_TOKENS * VGGT_FULL_TOKEN_EMBED_DIM


def flatten_world_points_camera_pose(out: dict) -> jnp.ndarray:
    """Flatten structured VGGT world-points + camera-pose for MLP encoders."""
    wp = out[WORLD_POINTS_KEY].reshape(-1)
    cp = out[CAMERA_POSE_KEY].reshape(-1)
    return jnp.concatenate([wp, cp]).astype(jnp.float32)


def structured_world_points_camera_pose(
    out: dict,
    *,
    expected_hwc_shape: tuple[int, int, int],
    world_points_key: str = "world_points",
) -> dict[str, jnp.ndarray]:
    """Return VGGT world points and camera pose as explicit fields.

    ``expected_hwc_shape`` is derived from the Encoder Input Contract. The
    adapter validates against the contract rather than owning VGGT dimensions.
    """
    world_points = out[world_points_key]
    if tuple(world_points.shape) != expected_hwc_shape:
        raise ValueError(
            f"expected {world_points_key} shape {expected_hwc_shape}, "
            f"got {tuple(world_points.shape)}"
        )
    return {
        WORLD_POINTS_KEY: jnp.transpose(world_points, (2, 0, 1)).astype(jnp.float32),
        CAMERA_POSE_KEY: out["camera_pose"].astype(jnp.float32),
    }


def _checked_feature(out: dict, key: str, expected_shape: tuple[int, ...]) -> jnp.ndarray:
    """Return a float32 VGGT readout after validating its contract shape."""
    features = out[key]
    if features.shape != expected_shape:
        raise ValueError(f"expected {key} shape {expected_shape}, got {features.shape}")
    return features.astype(jnp.float32)


def pool_aggregator_tokens(out: dict, expected_shape: tuple[int, ...]) -> jnp.ndarray:
    """Return three pre-head pools concatenated as a single (3*D,) vector.

    Layout matches the aggregator's ``[camera, register, patches]`` ordering
    (see ``src/vggt/jax/aggregator.py``): the camera token at index 0 is
    the embedding VGGT's own ``camera_head`` reads to predict pose, so we keep
    it unmixed. Register tokens (1:5) are attention sinks and dropped. Patches
    (5:) are reduced with both mean (smooth global) and max (salient features)
    so the encoder sees signals at different scales.
    """
    features = _checked_feature(out, "aggregator_features", expected_shape)
    cam = features[0]
    patches = features[VGGT_AGGREGATOR_PATCH_START_IDX:]
    mean_p = patches.mean(axis=0)
    max_p = patches.max(axis=0)
    return jnp.concatenate([cam, mean_p, max_p], axis=0)


def flatten_raw_aggregator(out: dict, expected_shape: tuple[int, ...]) -> jnp.ndarray:
    """Flatten the RAW aggregator tokens, dropping the 4 register tokens (JAX).

    Unlike ``pool_aggregator_tokens`` (which collapses patches to mean+max), this
    keeps every token: camera token (idx 0) + all patch tokens (idx
    ``VGGT_AGGREGATOR_PATCH_START_IDX:``) = ``1 + (P-5)`` tokens, flattened
    row-major to ``(n_tokens * embed_dim,)`` — 1370*1024 = 1,402,880 at the
    default config. The 4 register tokens (idx 1:5) are attention sinks and dropped, matching the
    pooled path. Stored float16 in replay (~2.81 MB/frame).
    """
    features = _checked_feature(out, "aggregator_features", expected_shape)
    cam = features[0:1]                          # (1, D)
    patches = features[VGGT_AGGREGATOR_PATCH_START_IDX:]  # (P-5, D)
    kept = jnp.concatenate([cam, patches], axis=0)  # (1 + P-5, D)
    return kept.reshape(-1)                       # (n_tokens * D,)


def flatten_full_aggregator_tokens(out: dict, expected_shape: tuple[int, ...]) -> jnp.ndarray:
    """Flatten all VGGT aggregator tokens, keeping camera, register, and patches.

    This is the 3D-75 token-Transformer replay layout. It intentionally differs
    from ``flatten_raw_aggregator`` by preserving register tokens so the trainable
    encoder sees the full frozen VGGT token sequence: ``(1374, 1024)`` at the
    default 518px / 37x37-patch configuration. Replay stores the flattened vector
    as float16; the Flax encoder upcasts before attention.
    """
    return _checked_feature(out, "aggregator_features", expected_shape).reshape(-1)


def full_aggregator_tokens(out: dict, expected_shape: tuple[int, ...]) -> jnp.ndarray:
    """Return full-width VGGT aggregator tokens for the 3D-77 context path."""
    return _checked_feature(out, "aggregator_full_tokens", expected_shape)


TOKEN_READOUTS = {
    "aggregator": pool_aggregator_tokens,
    "agg_raw": flatten_raw_aggregator,
    "agg_tokens": flatten_full_aggregator_tokens,
}


class VGGTObsAdapter(ObsAdapter):
    """Runs VGGT extraction, returns features for both buffer and agent."""

    def __init__(
        self,
        extractor: VGGTFeatureExtractor,
        feature_kind: VGGTFeatureKind = "wp_cp",
        *,
        env_render_resolution: int | None = None,
        encoder_type: str | None = None,
        encoder_module_cls=None,
        agent_overrides=None,
        design_notes: str = "",
    ):
        self.contract = build_vggt_contract(
            extractor,
            feature_kind=feature_kind,
            env_render_resolution=env_render_resolution,
            encoder_type=encoder_type,
            encoder_module_cls=encoder_module_cls,
            agent_overrides=agent_overrides,
            design_notes=design_notes,
        )
        super().__init__(
            buffer_dtype=self.contract.replay_observation.buffer_dtype(),
            buffer_shape=self.contract.replay_observation.buffer_shape(),
            normalize_on_sample=self.contract.replay_observation.buffer_normalize(),
            agent_obs_shape=self.contract.encoder_input.buffer_shape(),
            on_episode_reset=extractor.reset,
        )
        self._extractor = extractor
        self._feature_kind: VGGTFeatureKind = feature_kind
        self._aggregator_feature_shape = tuple(getattr(extractor, "aggregator_feature_shape", ()))

    def _extract(self, image: np.ndarray) -> dict:
        spec = head_readout_spec(self._feature_kind)
        if spec is not None and spec.use_dense_world_points:
            return self._extractor.extract(image, return_dense=True)
        return self._extractor.extract(image)

    def _readout(self, out: dict) -> jnp.ndarray | dict[str, jnp.ndarray]:
        if spec := head_readout_spec(self._feature_kind):
            return structured_world_points_camera_pose(
                out,
                expected_hwc_shape=contract_world_points_hwc_shape(self.contract),
                world_points_key=(
                    "dense_world_points" if spec.use_dense_world_points else "world_points"
                ),
            )
        if self._feature_kind in TOKEN_READOUTS:
            return TOKEN_READOUTS[self._feature_kind](out, self._aggregator_feature_shape)
        raise ValueError(f"unknown VGGT feature_kind {self._feature_kind!r}")

    def _to_replay_agent_obs(
        self,
        features_jax: jnp.ndarray | dict[str, jnp.ndarray],
        *,
        is_first: bool,
    ) -> tuple[np.ndarray | dict[str, np.ndarray], dict]:
        # The replay buffer is CPU/NumPy storage. The acting path keeps JAX
        # float32 features so it can feed the JIT-compiled agent directly.
        replay_dtype = self.contract.replay_observation.buffer_dtype()
        if isinstance(features_jax, dict):
            if not isinstance(replay_dtype, dict):
                raise ValueError("structured VGGT readouts require per-field replay dtypes")
            replay_features = {
                WORLD_POINTS_KEY: np.asarray(
                    features_jax[WORLD_POINTS_KEY],
                    dtype=np.dtype(replay_dtype[WORLD_POINTS_KEY]),
                ),
                CAMERA_POSE_KEY: np.asarray(
                    features_jax[CAMERA_POSE_KEY],
                    dtype=np.dtype(replay_dtype[CAMERA_POSE_KEY]),
                ),
            }
            agent_obs = {
                WORLD_POINTS_KEY: features_jax[WORLD_POINTS_KEY].astype(jnp.float32),
                CAMERA_POSE_KEY: features_jax[CAMERA_POSE_KEY].astype(jnp.float32),
                "is_first": is_first,
            }
            return replay_features, agent_obs

        if isinstance(replay_dtype, dict):
            raise ValueError("flat VGGT readouts require a scalar replay dtype")
        replay_features = np.asarray(features_jax, dtype=np.dtype(replay_dtype))
        agent_features = features_jax.astype(jnp.float32)
        agent_obs = {"features": agent_features, "is_first": is_first}
        return replay_features, agent_obs

    def transform(self, obs_dict: dict) -> tuple[np.ndarray | dict[str, np.ndarray], dict]:
        out = self._extract(obs_dict["image"])
        features_jax = self._readout(out)
        return self._to_replay_agent_obs(
            features_jax,
            is_first=obs_dict.get("is_first", False),
        )
