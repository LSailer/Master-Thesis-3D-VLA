"""R2DreamerAgent — composition root.

The agent is a thin orchestrator: it owns parameters, the LaProp optimizer,
the slow-target EMA, the acting state, and the JIT'd train/eval entry points.
The actual loss math lives in three subpackages, each with its own loss file:

    world_model/loss.py     — KL (dyn + rep) + reward + continue heads
    behavior/loss.py        — imagination rollout, actor + critic losses
    representation/loss.py  — Barlow Twins + replay-based value learning

A single shared forward pass (`_world_model_forward`) computes `embed`, the
RSSM posterior states, prior logits, and `feat`. Those tensors thread into
all three sub-loss functions so the encoder/RSSM receive the correct combined
gradient signal under one `jax.grad`.
"""

import functools
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, NamedTuple, cast

import jax
import jax.numpy as jnp
import optax

from src.buffer import ReplayBatch
from src.configs.config import R2DreamerConfig
from src.r2dreamer.decoder_targets import decoder_rgb_target, replay_batch_shape
from src.r2dreamer.encoders.shape_utils import batch_live_observation
from src.r2dreamer.observation_keys import (
    CAMERA_POSE_KEY,
    CAMERA_TOKEN_GLOBAL_KEY,
    FULL_TOKENS_KEY,
    GLOBAL_PATCH_TOKENS_KEY,
    GLOBAL_TOKENS_KEY,
    HOUSE_CONTEXT_KEY,
    HOUSE_CONTEXT_SIZE_KEY,
    HYBRID_IMAGE_KEY,
    WORLD_POINTS_KEY,
)
from src.shared.dtypes import compute_jnp_dtype
from src.shared.optim import agc, laprop

from .behavior.imagination import _imagine, _lambda_return
from .behavior.loss import behavior_loss
from .behavior.return_ema import ReturnEMA
from .checkpointing import load_checkpoint
from .encoders.cnn import ConvEncoder
from .encoders.constants import AGG_REGISTER_TOKENS, HYBRID_RGB_DIM
from .encoders.decoder import ConvDecoder
from .encoders.mlp import (
    HouseGlobalEmbeddingEncoder as WMHouseGlobalEmbeddingEncoder,
)
from .encoders.mlp import (
    HousePointsCameraEncoder,
    HybridHousePointsCameraEncoder,
    WP64CNNCPMLPEncoder,
)
from .encoders.mlp import (
    HybridEncoder as WMHybridEncoder,
)
from .encoders.pointnet import PointNetHousePointsCameraEncoder
from .encoders.mlp import (
    MLPEncoder as WMMLPEncoder,
)
from .encoders.mlp import (
    VGGTAggRawMLPEncoder as WMVGGTAggRawMLPEncoder,
)
from .encoders.mlp import (
    VGGTAggregatorMLPEncoder as WMVGGTAggregatorMLPEncoder,
)
from .encoders.transformer import TokenTransformerEncoder as WMTokenTransformerEncoder
from .learning_types import AgentLossAux, WorldModelForward
from .observation_preparation.contracts import (
    normalize_encoder_module_kwargs,
    recover_encoder_input_contract,
)
from .representation.barlow import Projector
from .representation.loss import representation_loss
from .world_model.heads import R2MLP, R2TwoHotDist
from .world_model.loss import kl_loss as _kl_loss
from .world_model.loss import world_model_loss
from .world_model.rssm import R2RSSM


class ActState(NamedTuple):
    """Functional single-env acting state."""

    stoch: jax.Array
    deter: jax.Array
    prev_action: jax.Array


class R2DTrainState(NamedTuple):
    """Mutable-on-Python-side training state bundled for the JIT step.

    The public ``R2DreamerAgent`` keeps exposing ``params``/``opt_state``/
    ``slow_critic_params``/``ema_state`` properties for checkpoint and test
    compatibility, but the compiled training kernel receives and returns this
    single pytree so its interface stays compact.
    """

    params: Dict[str, Any]
    opt_state: optax.OptState
    slow_critic_params: Dict[str, Any]
    ema_state: Any


# ---------------------------------------------------------------------------
# Module factories
# ---------------------------------------------------------------------------


def _compute_dtype_kwargs(cfg: R2DreamerConfig) -> dict[str, Any]:
    """Return the ``compute_dtype`` override for the ``full_bf16`` gate.

    Only supplies ``compute_dtype`` when ``cfg.full_bf16`` is set, so that
    with the gate off each module keeps its own default — historically
    float32 for the CNN/house/pose/RSSM/head path, but bfloat16 for modules
    that already opted in on their own (e.g. the PointNet house branch).

    Args:
      cfg: Agent config supplying ``full_bf16`` and ``compute_dtype``.

    Returns:
      ``{"compute_dtype": <jnp dtype>}`` when the gate is on, else ``{}``.
    """
    if getattr(cfg, "full_bf16", False):
        return {"compute_dtype": compute_jnp_dtype(cfg.compute_dtype)}
    return {}


def _make_rssm(cfg: R2DreamerConfig) -> R2RSSM:
    return R2RSSM(
        deter_size=cfg.deter_size,
        stoch_classes=cfg.stoch_classes,
        stoch_discrete=cfg.stoch_discrete,
        num_actions=cfg.num_actions,
        hidden=cfg.hidden_size,
        blocks=cfg.blocks,
        dyn_layers=cfg.dyn_layers,
        obs_layers=cfg.obs_layers,
        img_layers=cfg.img_layers,
        unimix_ratio=cfg.unimix_ratio,
        **_compute_dtype_kwargs(cfg),
    )


def load_policy_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load an R2DreamerAgent checkpoint, tolerating moved optimizer classes."""
    path = Path(path)
    ckpt = load_checkpoint(str(path))
    missing = {"params", "slow_critic_params"} - set(ckpt)
    if missing:
        raise KeyError(f"checkpoint {path} is missing required keys: {sorted(missing)}")
    return ckpt


def _resolve_encoder_cls(cfg: R2DreamerConfig):
    # Launcher-created configs pass EncoderSpec.module_cls explicitly. Unit tests
    # and direct R2DreamerConfig() construction rely on encoder_type, so map the
    # documented names to their Flax modules when no class is supplied.
    cls = cfg.encoder_module_cls
    if cls is None:
        cls = {
            "cnn": ConvEncoder,
            "vggt": WMMLPEncoder,
            "vggt_wp_cp_64": WMMLPEncoder,  # same MLP module, finer WP grid (obs 12297)
            "vggt_aggregator_mlp": WMVGGTAggregatorMLPEncoder,
            "vggt_agg_raw": WMVGGTAggRawMLPEncoder,
            "vggt_agg_token_transformer": WMTokenTransformerEncoder,
            "vggt_wp_dense_cnn": ConvEncoder,
            "vggt_wp64_cnn_cp_mlp": WP64CNNCPMLPEncoder,
            "hybrid": WMHybridEncoder,
            "vggt_house_context": WMHybridEncoder,
            "vggt_house_points_pose": PointNetHousePointsCameraEncoder,
            "pointnet": PointNetHousePointsCameraEncoder,
            "vggt_hybrid_house_points_pose": HybridHousePointsCameraEncoder,
            "vggt_house_full_tokens_nogate": WMTokenTransformerEncoder,
            "vggt_house_global_tokens_nogate": WMTokenTransformerEncoder,
            "vggt_house_global_embedding": WMHouseGlobalEmbeddingEncoder,
        }.get(cfg.encoder_type)
        if cls is None:
            raise ValueError(f"unknown encoder_type {cfg.encoder_type!r}")
    return cls


def _validate_encoder_config(cfg: R2DreamerConfig, cls) -> None:
    if cls in (ConvEncoder, WP64CNNCPMLPEncoder) and cfg.vggt_mlp_layers != 1:
        # Fail loud instead of silently dropping the knob: conv encoders have no
        # MLP depth, so a non-default vggt_mlp_layers here is a misconfiguration.
        raise ValueError(
            f"vggt_mlp_layers={cfg.vggt_mlp_layers} has no effect on "
            f"{cls.__name__} (a conv encoder, no MLP blocks). Only the 'vggt' and "
            f"'vggt_aggregator_mlp' encoders consume vggt_mlp_layers; leave it at 1 "
            f"for cnn / vggt_wp_dense_cnn."
        )


def _contract_encoder_kwargs(cfg: R2DreamerConfig) -> dict[str, Any]:
    snapshot = getattr(cfg, "encoder_input_contract", None)
    if snapshot is None:
        return {}
    return normalize_encoder_module_kwargs(snapshot.get("encoder_module_kwargs", {}))


def _make_conv_encoder(cfg: R2DreamerConfig):
    kwargs = _contract_encoder_kwargs(cfg)
    if kwargs:
        return ConvEncoder(**kwargs)
    return ConvEncoder(
        depth=cfg.encoder_depth,
        kernel_size=cfg.encoder_kernel,
        mults=cfg.encoder_mults,
        **_compute_dtype_kwargs(cfg),
    )


def _make_wp_conv_encoder(cfg: R2DreamerConfig):
    # Full-res world-point map -> conv stack -> embed_dim (3D-53). Reuses the
    # RGB conv hyperparameters; symlog (not /255) handles the metric XYZ range.
    kwargs = _contract_encoder_kwargs(cfg)
    if kwargs:
        return ConvEncoder(**kwargs)
    return ConvEncoder(
        input_kind="world_points",
        embed_dim=cfg.vggt_embed_dim,
        depth=cfg.encoder_depth,
        kernel_size=cfg.encoder_kernel,
        mults=cfg.encoder_mults,
    )


def _make_wp64_cnn_cp_mlp_encoder(cfg: R2DreamerConfig):
    kwargs = _contract_encoder_kwargs(cfg)
    if kwargs:
        return WP64CNNCPMLPEncoder(**kwargs)
    return WP64CNNCPMLPEncoder(
        embed_dim=cfg.vggt_embed_dim,
        conv_depth=cfg.encoder_depth,
        conv_kernel=cfg.encoder_kernel,
        conv_mults=cfg.encoder_mults,
        cp_hidden=cfg.mlp_vggt_hidden,
        cp_layers=cfg.mlp_vggt_layers,
    )


def _make_hybrid_encoder(cfg: R2DreamerConfig):
    # CNN(RGB) + gated MLP(WP/CP) fused into one embed (3D-50/51/52).
    # HybridEncoder now owns structured replay/live layout handling.
    expected_shape = (HYBRID_RGB_DIM + cfg.vggt_feature_dim,)
    if not isinstance(cfg.obs_shape, tuple):
        raise ValueError(f"hybrid expects flat obs_shape, got {cfg.obs_shape}")
    if not (
        cfg.obs_shape == expected_shape
        and cfg.obs_shape[0] - cfg.vggt_feature_dim == HYBRID_RGB_DIM
    ):
        raise ValueError(
            "hybrid obs_shape/split mismatch: expected "
            f"{expected_shape} with vggt_feature_dim={cfg.vggt_feature_dim}, "
            f"got obs_shape={cfg.obs_shape}, "
            f"vggt_feature_dim={cfg.vggt_feature_dim}"
        )
    kwargs = _contract_encoder_kwargs(cfg)
    if kwargs:
        return WMHybridEncoder(**kwargs)
    return WMHybridEncoder(
        cnn_depth=cfg.encoder_depth,
        cnn_kernel=cfg.encoder_kernel,
        cnn_mults=cfg.encoder_mults,
        vggt_embed_dim=cfg.vggt_embed_dim,
        mlp_hidden=cfg.mlp_vggt_hidden,
        mlp_layers=cfg.mlp_vggt_layers,
        vggt_dim=cfg.vggt_feature_dim,
    )


def _make_house_points_camera_encoder(
    cfg: R2DreamerConfig, cls: type = HousePointsCameraEncoder
):
    kwargs = _contract_encoder_kwargs(cfg)
    if kwargs:
        return cls(**kwargs)
    kwargs = dict(
        embed_dim=cfg.vggt_embed_dim,
        camera_hidden=cfg.mlp_vggt_hidden,
        camera_layers=cfg.mlp_vggt_layers,
        point_hidden=cfg.mlp_vggt_hidden,
        point_layers=cfg.mlp_vggt_layers,
        house_point_norm=cfg.house_point_norm,
        **_compute_dtype_kwargs(cfg),
    )
    if issubclass(cls, HybridHousePointsCameraEncoder):
        kwargs.update(
            cnn_depth=cfg.encoder_depth,
            cnn_kernel=cfg.encoder_kernel,
            cnn_mults=cfg.encoder_mults,
        )
    return cls(**kwargs)


def _make_house_global_embedding_encoder(
    cfg: R2DreamerConfig, cls: type = WMHouseGlobalEmbeddingEncoder
):
    # PointNet reducer over VGGT global patch tokens + camera side branch.
    # token_dim and num_patch_tokens are fixed by the VGGT global-half token
    # layout (camera token + 4 registers dropped): num_patch_tokens =
    # vggt_token_count - (1 camera + AGG_REGISTER_TOKENS). Prod sets
    # vggt_token_dim=1024 / vggt_token_count=1374 via agent_overrides; tests
    # may inject small dims.
    kwargs = _contract_encoder_kwargs(cfg)
    if kwargs:
        return cls(**kwargs)
    num_patch_tokens = int(cfg.vggt_token_count) - (1 + AGG_REGISTER_TOKENS)
    return cls(
        embed_dim=cfg.vggt_embed_dim,
        token_dim=cfg.vggt_token_dim,
        num_patch_tokens=num_patch_tokens,
        reducer_hidden=cfg.mlp_vggt_hidden,
        reducer_layers=cfg.mlp_vggt_layers,
        camera_hidden=cfg.mlp_vggt_hidden,
        camera_layers=cfg.mlp_vggt_layers,
    )


def _make_mlp_encoder(cfg: R2DreamerConfig, cls):
    # wp_cp + aggregator MLP encoders: depth from cfg.vggt_mlp_layers (3D-52).
    kwargs = _contract_encoder_kwargs(cfg)
    if kwargs:
        return cls(**kwargs)
    return cls(
        embed_dim=cfg.vggt_embed_dim,
        hidden=cfg.vggt_embed_dim,
        num_layers=cfg.vggt_mlp_layers,
    )


def _make_rgb_token_encoder(cfg: R2DreamerConfig):
    token_key = FULL_TOKENS_KEY
    singleton_tokens = False
    if cfg.encoder_type == "vggt_house_global_tokens_nogate":
        token_key = GLOBAL_TOKENS_KEY
        singleton_tokens = True
    return WMTokenTransformerEncoder(
        embed_dim=cfg.vggt_embed_dim,
        token_dim=cfg.vggt_token_dim,
        num_tokens=cfg.vggt_token_count,
        model_dim=None,
        layers=cfg.vggt_token_transformer_layers,
        heads=cfg.vggt_token_transformer_heads,
        mlp_ratio=cfg.vggt_token_transformer_mlp_ratio,
        dropout=cfg.vggt_token_transformer_dropout,
        readout="mean",
        norm_kind="layer",
        activation="gelu",
        token_key=token_key,
        image_key=HYBRID_IMAGE_KEY,
        singleton_tokens=singleton_tokens,
        compute_dtype=compute_jnp_dtype(cfg.compute_dtype),
        cnn_depth=cfg.encoder_depth,
        cnn_kernel=cfg.encoder_kernel,
        cnn_mults=cfg.encoder_mults,
    )


def _make_token_transformer_encoder(cfg: R2DreamerConfig):
    return WMTokenTransformerEncoder(
        embed_dim=cfg.vggt_embed_dim,
        token_dim=cfg.vggt_token_dim,
        num_tokens=cfg.vggt_token_count,
        model_dim=cfg.vggt_token_projection_dim,
        layers=cfg.vggt_token_transformer_layers,
        heads=cfg.vggt_token_transformer_heads,
        mlp_ratio=cfg.vggt_token_transformer_mlp_ratio,
        readout="camera_register_patch",
        norm_kind="rms",
        activation="silu",
        keep_register_tokens=cfg.vggt_keep_register_tokens,
        compute_dtype=compute_jnp_dtype(cfg.compute_dtype),
    )


def _make_encoder(cfg: R2DreamerConfig):
    cls = _resolve_encoder_cls(cfg)
    _validate_encoder_config(cfg, cls)
    if cls is ConvEncoder:
        if cfg.encoder_type == "vggt_wp_dense_cnn":
            return _make_wp_conv_encoder(cfg)
        return _make_conv_encoder(cfg)
    if cls is WP64CNNCPMLPEncoder:
        return _make_wp64_cnn_cp_mlp_encoder(cfg)
    if cls is WMHybridEncoder:
        return _make_hybrid_encoder(cfg)
    if cls is WMHouseGlobalEmbeddingEncoder:
        return _make_house_global_embedding_encoder(cfg, cls)
    if issubclass(cls, HousePointsCameraEncoder):
        # issubclass so GNN variants (src/r2dreamer/encoders/gnn_house.py)
        # reuse this builder with their own module class.
        return _make_house_points_camera_encoder(cfg, cls)
    if cls is WMTokenTransformerEncoder:
        if cfg.encoder_type == "vggt_agg_token_transformer":
            return _make_token_transformer_encoder(cfg)
        return _make_rgb_token_encoder(cfg)
    return _make_mlp_encoder(cfg, cls)


def _dummy_encoder_obs(cfg: R2DreamerConfig):
    if cfg.encoder_type == "vggt_house_full_tokens_nogate":
        return {
            HYBRID_IMAGE_KEY: jnp.zeros((1, 3, 64, 64), dtype=jnp.float32),
            FULL_TOKENS_KEY: jnp.zeros(
                (1, cfg.vggt_token_count, cfg.vggt_token_dim),
                dtype=compute_jnp_dtype(cfg.compute_dtype),
            ),
        }
    if cfg.encoder_type == "vggt_house_global_tokens_nogate":
        return {
            HYBRID_IMAGE_KEY: jnp.zeros((1, 3, 64, 64), dtype=jnp.float32),
            GLOBAL_TOKENS_KEY: jnp.zeros(
                (1, cfg.vggt_token_count, cfg.vggt_token_dim),
                dtype=compute_jnp_dtype(cfg.compute_dtype),
            ),
        }
    if cfg.encoder_type == "vggt_house_global_embedding":
        if not isinstance(cfg.obs_shape, Mapping):
            raise TypeError(f"{cfg.encoder_type} expects structured obs_shape")
        return {
            HYBRID_IMAGE_KEY: jnp.zeros(
                (1, *cfg.obs_shape[HYBRID_IMAGE_KEY]), dtype=jnp.float32
            ),
            CAMERA_TOKEN_GLOBAL_KEY: jnp.zeros(
                (1, *cfg.obs_shape[CAMERA_TOKEN_GLOBAL_KEY]), dtype=jnp.float32
            ),
            GLOBAL_PATCH_TOKENS_KEY: jnp.zeros(
                (1, *cfg.obs_shape[GLOBAL_PATCH_TOKENS_KEY]), dtype=jnp.float32
            ),
        }
    if cfg.encoder_type == "vggt_wp64_cnn_cp_mlp":
        return {
            WORLD_POINTS_KEY: jnp.zeros((1, 3, 64, 64), dtype=jnp.float32),
            CAMERA_POSE_KEY: jnp.zeros((1, 9), dtype=jnp.float32),
        }
    if cfg.encoder_type in (
        "vggt_house_points_pose",
        "vggt_hybrid_house_points_pose",
        "gnn_house_points_pose",
        "gnn_edge_house_points_pose",
        "pointnet",
    ):
        if not isinstance(cfg.obs_shape, Mapping):
            raise TypeError(f"{cfg.encoder_type} expects structured obs_shape")
        dummy = {
            CAMERA_POSE_KEY: jnp.zeros((1, 9), dtype=jnp.float32),
            HOUSE_CONTEXT_KEY: jnp.zeros(
                (1, *cfg.obs_shape[HOUSE_CONTEXT_KEY]), dtype=jnp.float32
            ),
            HOUSE_CONTEXT_SIZE_KEY: jnp.zeros((), dtype=jnp.int32),
        }
        if cfg.encoder_type == "vggt_hybrid_house_points_pose":
            dummy[HYBRID_IMAGE_KEY] = jnp.zeros((1, 3, 64, 64), dtype=jnp.float32)
        return dummy
    return jnp.zeros((1, *cfg.obs_shape))


def _weighted_total_loss(cfg: R2DreamerConfig, losses: dict[str, Any]):
    """Agent objective, excluding the optional debug decoder probe."""
    return (
        cfg.scale_dyn * losses["dyn"]
        + cfg.scale_rep * losses["rep"]
        + cfg.scale_barlow * losses["barlow"]
        + cfg.scale_rew * losses["rew"]
        + cfg.scale_con * losses["con"]
        + cfg.scale_policy * losses["policy"]
        + cfg.scale_value * losses["value"]
        + cfg.scale_repval * losses["repval"]
    )


def _add_loss_metrics(metrics: dict[str, Any], losses: dict[str, Any]) -> None:
    for k, v in losses.items():
        metrics[f"loss/{k}"] = v


def _add_encoder_l2_metric(metrics: dict[str, Any], params: dict[str, Any]) -> None:
    # Encoder L2 — Protocol D diagnostic for whether Barlow's gradient
    # toggle is actually moving the encoder weights.
    enc_sq = jax.tree_util.tree_reduce(
        lambda acc, x: acc + jnp.sum(jnp.square(x)),
        params["encoder"],
        0.0,
    )
    metrics["params/encoder_l2"] = jnp.sqrt(enc_sq)


def _add_hybrid_contribution_metrics(
    metrics: dict[str, Any],
    *,
    cfg: R2DreamerConfig,
    params: dict[str, Any],
    forward: WorldModelForward,
    B: int,
    T: int,
) -> None:
    # Reuse the already-computed fused embed instead of a second encoder
    # forward: embed == concat([cnn_e, gate * vggt_mlp(...)]), so the
    # leading cnn_dim columns are the CNN branch and the rest are the
    # gated VGGT branch. The raw gate scalar is read straight from params.
    embed_flat = forward.embed.reshape(B * T, -1)
    cnn_dim = embed_flat.shape[-1] - cfg.vggt_embed_dim
    cnn_e = embed_flat[:, :cnn_dim]
    vggt_e = embed_flat[:, cnn_dim:]
    gate = params["encoder"]["params"]["gate"]
    cnn_l2 = jnp.sqrt(jnp.mean(jnp.sum(cnn_e**2, axis=-1)))
    vggt_l2 = jnp.sqrt(jnp.mean(jnp.sum(vggt_e**2, axis=-1)))
    denom = cnn_l2 + vggt_l2 + 1e-8
    metrics["hybrid/gate"] = gate
    metrics["hybrid/cnn_l2"] = cnn_l2
    metrics["hybrid/vggt_l2"] = vggt_l2
    metrics["hybrid/cnn_std"] = jnp.std(cnn_e)
    metrics["hybrid/vggt_std"] = jnp.std(vggt_e)
    metrics["hybrid/cnn_frac"] = cnn_l2 / denom
    metrics["hybrid/vggt_frac"] = vggt_l2 / denom


# ---------------------------------------------------------------------------
# R2DreamerAgent
# ---------------------------------------------------------------------------


class R2DreamerAgent:
    """R2-Dreamer agent with a single LaProp optimizer over all parameters.

    All Flax modules are *stateless* — parameters live in a flat pytree dict
    exposed as ``self.params``.  Training state is bundled in
    ``self.train_state`` and threaded through one JIT-compiled pure step.
    """

    @property
    def train_state(self) -> R2DTrainState:
        return self._train_state

    @train_state.setter
    def train_state(self, state: R2DTrainState) -> None:
        self._train_state = state

    @property
    def params(self):
        return self._train_state.params

    @params.setter
    def params(self, params):
        self._train_state = self._train_state._replace(params=params)

    @property
    def opt_state(self):
        return self._train_state.opt_state

    @opt_state.setter
    def opt_state(self, opt_state):
        self._train_state = self._train_state._replace(opt_state=opt_state)

    @property
    def slow_critic_params(self):
        return self._train_state.slow_critic_params

    @slow_critic_params.setter
    def slow_critic_params(self, slow_critic_params):
        self._train_state = self._train_state._replace(
            slow_critic_params=slow_critic_params
        )

    @property
    def ema_state(self):
        return self._train_state.ema_state

    @ema_state.setter
    def ema_state(self, ema_state):
        self._train_state = self._train_state._replace(ema_state=ema_state)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        obs_shape: tuple[int, ...] | dict[str, tuple[int, ...]] | None = None,
        num_actions: int,
        seed: int,
        **config_kwargs: Any,
    ) -> "R2DreamerAgent":
        """Build an agent and load ``params`` + ``slow_critic_params`` from disk.

        Extra ``config_kwargs`` flow into :class:`R2DreamerConfig` so callers
        that need ``encoder_type`` / ``encoder_module_cls`` (e.g. evaluate)
        can pass them through. When the checkpoint contains a durable Encoder
        Input Contract snapshot, missing encoder config is recovered from it.
        The loaded checkpoint's ``step`` is stashed on the returned agent as
        ``checkpoint_step`` (``-1`` if absent).
        """
        ckpt = load_policy_checkpoint(path)
        contract_snapshot = ckpt.get("encoder_input_contract")
        if contract_snapshot is not None:
            contract = recover_encoder_input_contract(contract_snapshot)
            if obs_shape is None:
                obs_shape = contract.encoder_input.buffer_shape()
            requested_type = config_kwargs.get("encoder_type")
            if requested_type is not None and requested_type != contract.encoder_type:
                raise ValueError(
                    "checkpoint encoder contract mismatch: requested "
                    f"{requested_type!r}, checkpoint has {contract.encoder_type!r}"
                )
            requested_shape = obs_shape
            contract_shape = contract.encoder_input.buffer_shape()
            if requested_shape != contract_shape:
                raise ValueError(
                    "checkpoint encoder shape mismatch: requested "
                    f"{requested_shape!r}, checkpoint has {contract_shape!r}"
                )
            config_kwargs["encoder_type"] = contract.encoder_type
            config_kwargs["encoder_module_cls"] = contract.encoder_module_cls
            config_kwargs["encoder_input_contract"] = contract_snapshot
        if obs_shape is None:
            raise ValueError(
                "obs_shape must be provided when checkpoint has no Encoder Input "
                "Contract snapshot"
            )
        config = R2DreamerConfig(
            obs_shape=obs_shape,
            num_actions=num_actions,
            **config_kwargs,
        )
        rng_key = jax.random.PRNGKey(seed)
        rng_key, init_key = jax.random.split(rng_key)
        agent = cls(config, init_key)
        agent.params = jax.tree.map(jnp.asarray, ckpt["params"])
        agent.slow_critic_params = jax.tree.map(jnp.asarray, ckpt["slow_critic_params"])
        agent.checkpoint_step = int(ckpt.get("step", -1))
        return agent

    def __init__(self, config: R2DreamerConfig, rng_key: jnp.ndarray):
        self.cfg = config
        self.checkpoint_step = -1
        self.twohot = R2TwoHotDist(num_bins=config.twohot_bins)

        # ---- Instantiate Flax modules (for .apply) ----
        self.encoder_mod = _make_encoder(config)
        self.rssm_mod = _make_rssm(config)

        # Dummy forward to discover embed_size
        rng_key, k1, k2, k3 = jax.random.split(rng_key, 4)
        dummy_obs = _dummy_encoder_obs(config)
        enc_params = self.encoder_mod.init(k1, dummy_obs)
        embed = cast(jnp.ndarray, self.encoder_mod.apply(enc_params, dummy_obs))
        self.embed_size = embed.shape[-1]

        # RSSM
        stoch0 = jnp.zeros((1, config.stoch_classes, config.stoch_discrete))
        deter0 = jnp.zeros((1, config.deter_size))
        action0 = jnp.zeros((1, config.num_actions))
        embed0 = jnp.zeros((1, self.embed_size))
        rng_key, k_sample = jax.random.split(rng_key)
        rssm_params = self.rssm_mod.init(
            {"params": k2, "sample": k_sample}, stoch0, deter0, action0, embed0
        )

        # Projector: feat_size -> embed_size
        self.proj_mod = Projector(out_dim=self.embed_size)
        feat0 = jnp.zeros((1, config.feat_size))
        proj_params = self.proj_mod.init(k3, feat0)

        # MLP heads (outscale matches PyTorch: 0.0 for reward/critic, 0.01 for actor)
        rng_key, k_rew, k_con, k_act, k_cri = jax.random.split(rng_key, 5)
        head_dtype_kwargs = _compute_dtype_kwargs(config)
        self.reward_mod = R2MLP(
            hidden=config.mlp_units,
            layers=config.mlp_layers_reward,
            out_dim=config.twohot_bins,
            outscale=0.0,
            **head_dtype_kwargs,
        )
        rew_params = self.reward_mod.init(k_rew, feat0)

        self.cont_mod = R2MLP(
            hidden=config.mlp_units,
            layers=config.mlp_layers_cont,
            out_dim=1,
            **head_dtype_kwargs,
        )
        con_params = self.cont_mod.init(k_con, feat0)

        self.actor_mod = R2MLP(
            hidden=config.mlp_units,
            layers=config.mlp_layers_actor,
            out_dim=config.num_actions,
            outscale=0.01,
            **head_dtype_kwargs,
        )
        act_params = self.actor_mod.init(k_act, feat0)

        self.critic_mod = R2MLP(
            hidden=config.mlp_units,
            layers=config.mlp_layers_critic,
            out_dim=config.twohot_bins,
            outscale=0.0,
            **head_dtype_kwargs,
        )
        cri_params = self.critic_mod.init(k_cri, feat0)

        # ---- Debug decoder probe (3D-51): built ONLY when cfg.decoder ----
        # Reconstructs RGB from stop-gradient `feat` for visual verification.
        # Left unbuilt by default so the params pytree (and thus checkpoints) of
        # CNN/VGGT runs is unchanged.
        self.decoder_mod = None
        dec_params = None
        if config.decoder:
            if config.encoder_type not in (
                "cnn",
                "hybrid",
                "vggt_house_context",
                "vggt_house_full_tokens_nogate",
                "vggt_house_global_tokens_nogate",
                "vggt_house_global_embedding",
            ):
                raise ValueError(
                    "decoder=True requires an RGB-bearing encoder_type — the "
                    "ConvDecoder reconstructs an RGB image, but "
                    f"{config.encoder_type!r} carries no RGB modality to reconstruct."
                )
            rng_key, k_dec = jax.random.split(rng_key)
            self.decoder_mod = ConvDecoder(
                depth=config.encoder_depth,
                kernel_size=config.encoder_kernel,
                mults=config.encoder_mults,
            )
            dec_params = self.decoder_mod.init(k_dec, feat0)

        # ---- Bundle all params ----
        params = {
            "encoder": enc_params,
            "rssm": rssm_params,
            "projector": proj_params,
            "reward": rew_params,
            "cont": con_params,
            "actor": act_params,
            "critic": cri_params,
        }

        # Module bundle passed to sub-loss functions
        self._modules = {
            "encoder": self.encoder_mod,
            "rssm": self.rssm_mod,
            "projector": self.proj_mod,
            "reward": self.reward_mod,
            "cont": self.cont_mod,
            "actor": self.actor_mod,
            "critic": self.critic_mod,
        }

        if config.decoder:
            params["decoder"] = dec_params
            self._modules["decoder"] = self.decoder_mod

        # ---- Optimizer: LaProp with linear warmup ----
        self.tx = laprop(
            lr=config.lr,
            b1=config.beta1,
            b2=config.beta2,
            eps=config.eps,
            warmup=config.warmup_steps,
        )
        opt_state = self.tx.init(params)

        # ---- Slow target critic (EMA) ----
        slow_critic_params = jax.tree.map(jnp.copy, params["critic"])

        # ---- Return EMA ----
        self.return_ema = ReturnEMA()
        ema_state = self.return_ema.init_state()

        self.train_state = R2DTrainState(
            params=params,
            opt_state=opt_state,
            slow_critic_params=slow_critic_params,
            ema_state=ema_state,
        )

        # ---- Acting state (for legacy single-env stepping wrapper) ----
        self._act_state = self.initial_act_state()

        # ---- JIT-compiled functions ----
        self._jitted_train_step = cast(Any, jax.jit(self._train_step_pure))
        self._jit_act_with_state = cast(Any, jax.jit(self.act_with_state_pure))

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def initial_act_state(self) -> ActState:
        """Return a zeroed functional single-env acting state."""
        return ActState(
            stoch=jnp.zeros(
                (1, self.cfg.stoch_classes, self.cfg.stoch_discrete), dtype=jnp.float32
            ),
            deter=jnp.zeros((1, self.cfg.deter_size), dtype=jnp.float32),
            prev_action=jnp.zeros((1, self.cfg.num_actions), dtype=jnp.float32),
        )

    def snapshot_act_state(self) -> ActState:
        """Copy the legacy mutable wrapper's acting state."""
        return jax.tree.map(jnp.copy, self._act_state)

    def restore_act_state(self, state: ActState) -> None:
        """Restore the legacy mutable wrapper's acting state."""
        self._act_state = state

    def act(
        self,
        encoder_obs: Any,
        is_first: bool,
        rng_key: jnp.ndarray,
        training: bool = True,
    ) -> int:
        """Select an action for a single prepared environment step.

        Args:
            encoder_obs: one live observation in the layout consumed by the encoder.
                The agent adds the single-env batch dimension internally.
            is_first: whether the step starts an episode and should reset RSSM state.
            rng_key: PRNG key.
            training: if False, use argmax (greedy).

        Returns:
            Integer action in [0, num_actions).
        """
        reset = jnp.asarray(is_first, dtype=jnp.bool_)
        batched_obs = batch_live_observation(encoder_obs)
        action_int, self._act_state = self._jit_act_with_state.__call__(
            self.params, batched_obs, self._act_state, reset, rng_key, training
        )

        # Honor the ``-> int`` contract: the jitted core returns a 0-d JAX array,
        # but callers (env.step, action_counts indexing) need a host Python int.
        # habitat's env.step only wraps int/np.integer into {"action": ...}; a
        # raw JAX array slips through to string indexing and raises.
        return int(action_int)

    def act_with_state(
        self,
        encoder_obs: Any,
        is_first: bool,
        state: ActState,
        rng_key: jnp.ndarray,
        training: bool = True,
    ) -> tuple[int, ActState]:
        """Functional acting wrapper for one raw live encoder observation."""
        reset = jnp.asarray(is_first, dtype=jnp.bool_)
        batched_obs = batch_live_observation(encoder_obs)
        action_int, new_state = self._jit_act_with_state.__call__(
            self.params, batched_obs, state, reset, rng_key, training
        )
        # As in ``act``: return a host int action (state pytree passes through
        # untouched so the next jitted call still sees stable shapes/dtypes).
        return int(action_int), new_state

    def act_with_state_pure(
        self, params, obs, state: ActState, is_first, rng_key, training
    ):
        """JIT-able acting logic. Returns (action_int, next ActState)."""
        state = jax.lax.cond(
            is_first,
            lambda _: self.initial_act_state(),
            lambda current: current,
            state,
        )
        embed = cast(jnp.ndarray, self.encoder_mod.apply(params["encoder"], obs))
        rng_key, k_sample = jax.random.split(rng_key)
        new_stoch, new_deter, _ = cast(
            tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
            self.rssm_mod.apply(
                params["rssm"],
                state.stoch,
                state.deter,
                state.prev_action,
                embed,
                rngs={"sample": k_sample},
            ),
        )
        feat = cast(
            jnp.ndarray,
            self.rssm_mod.apply(
                params["rssm"], new_stoch, new_deter, method=self.rssm_mod.get_feat
            ),
        )
        logits = self.actor_mod.apply(params["actor"], feat)

        def _sample(logits, rng_key):
            return jax.random.categorical(rng_key, logits, axis=-1)[0]

        def _greedy(logits, _rng_key):
            return jnp.argmax(logits, axis=-1)[0]

        action_int = jax.lax.cond(training, _sample, _greedy, logits, rng_key)
        new_state = ActState(
            stoch=new_stoch,
            deter=new_deter,
            prev_action=jax.nn.one_hot(
                action_int, self.cfg.num_actions, dtype=jnp.float32
            )[None],
        )
        return action_int, new_state

    # ------------------------------------------------------------------
    # Decoder reconstruction probe (visual verification; only when cfg.decoder)
    # ------------------------------------------------------------------

    def reconstruct(self, batch: Any):
        """Decode RGB reconstructions for a batch (encoder -> RSSM -> decoder).

        Returns ``(target, recon)`` as JAX arrays ``(B*T, 3, 64, 64)`` in
        [0, 1], or ``None`` when no decoder is configured. Non-JIT, deterministic
        (fixed sample key) — called by the trainer at log cadence for W&B image
        logging, so it is intentionally cheap-and-occasional rather than fast.
        """
        if not self.cfg.decoder or self.decoder_mod is None:
            return None
        params = self.params
        B, T = replay_batch_shape(batch)
        embed = cast(
            jnp.ndarray, self.encoder_mod.apply(params["encoder"], batch.obs)
        )
        if embed.shape[:2] != (B, T):
            raise ValueError(
                f"encoder must preserve replay leading dims {(B, T)}, got {embed.shape}"
            )
        stoch0, deter0 = cast(
            tuple[jnp.ndarray, jnp.ndarray],
            self.rssm_mod.apply(params["rssm"], B, method=self.rssm_mod.initial_state),
        )
        post_stochs, post_deters, _ = cast(
            tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
            self.rssm_mod.apply(
                params["rssm"],
                embed,
                batch.actions,
                (stoch0, deter0),
                batch.is_first,
                method=self.rssm_mod.observe,
                rngs={"sample": jax.random.PRNGKey(0)},
            ),
        )
        feat = cast(
            jnp.ndarray,
            self.rssm_mod.apply(
                params["rssm"], post_stochs, post_deters, method=self.rssm_mod.get_feat
            ),
        )
        recon = self.decoder_mod.apply(params["decoder"], feat.reshape(B * T, -1))
        target = decoder_rgb_target(batch, self.cfg.encoder_type)
        return target, recon

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(
        self,
        batch: ReplayBatch,
        rng_key: jnp.ndarray,
        *,
        materialize: bool = True,
    ) -> Dict[str, Any]:
        """One LaProp step on `batch`.

        Args:
          batch: The replay batch to train on.
          rng_key: PRNG key for the step.
          materialize: When ``True`` (default), block and return Python-float
            metrics. When ``False``, return the raw device-array metrics
            without forcing a device->host sync. The hot training loop passes
            ``False`` on non-logging steps so JAX async dispatch is not
            serialized ~every step for metrics that would be discarded.

        Returns:
          A dict of metric name to value. Python floats when ``materialize``,
          otherwise device ``jax.Array`` scalars.
        """
        self.train_state, metrics = self._jitted_train_step.__call__(
            self.train_state,
            batch,
            rng_key,
        )
        if materialize:
            return {k: float(v) for k, v in metrics.items()}
        return dict(metrics)

    def eval_loss(self, batch: Any, rng_key: jnp.ndarray) -> Dict[str, float]:
        """Evaluate the current objective on a batch without updating state."""
        _total_loss, aux = self._loss_fn(
            self.params,
            slow_critic_params=self.slow_critic_params,
            ema_state=self.ema_state,
            batch=batch,
            rng_key=rng_key,
        )
        metrics = dict(aux.metrics)
        metrics["total_loss"] = aux.agent_loss
        return {k: float(v) for k, v in metrics.items()}

    def _train_step_pure(self, state: R2DTrainState, batch, rng_key):
        """Pure-functional training step (JIT-able)."""
        params = state.params
        opt_state = state.opt_state
        slow_critic_params = state.slow_critic_params
        ema_state = state.ema_state

        # Slow critic EMA: update BEFORE loss (matches PyTorch _update_slow_target)
        tau = self.cfg.slow_target_fraction
        updated_slow = jax.tree.map(
            lambda s, p: tau * p + (1 - tau) * s,
            slow_critic_params,
            params["critic"],
        )

        loss_fn = functools.partial(
            self._loss_fn,
            slow_critic_params=updated_slow,
            ema_state=ema_state,
            batch=batch,
            rng_key=rng_key,
        )

        (total_loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        # NaN guard: skip update if loss is non-finite (mirrors PyTorch GradScaler)
        is_finite = jnp.isfinite(total_loss)

        grads = agc(grads, params, clip=self.cfg.agc_clip, pmin=self.cfg.agc_pmin)
        updates, new_opt_state = self.tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        new_ema_state = self.return_ema.update(ema_state, aux.imag_returns)

        # Roll back to pre-update state on NaN/inf
        new_params = jax.tree.map(
            lambda new, old: jnp.where(is_finite, new, old), new_params, params
        )
        new_opt_state = jax.tree.map(
            lambda new, old: jnp.where(is_finite, new, old), new_opt_state, opt_state
        )
        new_slow = jax.tree.map(
            lambda new, old: jnp.where(is_finite, new, old),
            updated_slow,
            slow_critic_params,
        )
        new_ema_state = jax.tree.map(
            lambda new, old: jnp.where(is_finite, new, old), new_ema_state, ema_state
        )

        metrics = aux.metrics
        metrics["opt_loss"] = total_loss
        metrics["total_loss"] = aux.agent_loss
        metrics["nan_skipped"] = 1.0 - is_finite.astype(jnp.float32)
        new_state = R2DTrainState(
            params=new_params,
            opt_state=new_opt_state,
            slow_critic_params=new_slow,
            ema_state=new_ema_state,
        )
        return new_state, metrics

    # ------------------------------------------------------------------
    # Composition root: shared forward + 3 sub-losses
    # ------------------------------------------------------------------

    def _world_model_forward(self, params, batch, rng_key) -> WorldModelForward:
        """Encoder + posterior rollout + prior + features. Shared across sub-losses.

        Computing this once is essential: if each sub-loss recomputed `embed`,
        the encoder would receive doubled gradient signal and the
        `barlow_stop_grad` toggle would no longer mean what it claims.
        """
        cfg = self.cfg
        B, T = replay_batch_shape(batch)

        embed = cast(
            jnp.ndarray, self.encoder_mod.apply(params["encoder"], batch.obs)
        )
        if embed.shape[:2] != (B, T):
            raise ValueError(
                f"encoder must preserve replay leading dims {(B, T)}, got {embed.shape}"
            )

        stoch0, deter0 = cast(
            tuple[jnp.ndarray, jnp.ndarray],
            self.rssm_mod.apply(params["rssm"], B, method=self.rssm_mod.initial_state),
        )

        rng_key, k_obs = jax.random.split(rng_key)
        post_stochs, post_deters, post_logits = cast(
            tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
            self.rssm_mod.apply(
                params["rssm"],
                embed,
                batch.actions,
                (stoch0, deter0),
                batch.is_first,
                method=self.rssm_mod.observe,
                rngs={"sample": k_obs},
            ),
        )

        rng_key, k_prior = jax.random.split(rng_key)
        _, prior_logits_flat = cast(
            tuple[jnp.ndarray, jnp.ndarray],
            self.rssm_mod.apply(
                params["rssm"],
                post_deters.reshape(B * T, -1),
                method=self.rssm_mod.prior,
                rngs={"sample": k_prior},
            ),
        )
        prior_logits = prior_logits_flat.reshape(
            B, T, cfg.stoch_classes, cfg.stoch_discrete
        )

        feat = cast(
            jnp.ndarray,
            self.rssm_mod.apply(
                params["rssm"],
                post_stochs,
                post_deters,
                method=self.rssm_mod.get_feat,
            ),
        )

        return WorldModelForward(
            embed=embed,
            post_stochs=post_stochs,
            post_deters=post_deters,
            post_logits=post_logits,
            prior_logits=prior_logits,
            feat=feat,
        )

    def _loss_fn(self, params, *, slow_critic_params, ema_state, batch, rng_key):
        """Compose the world-model, behavior, and representation losses.

        Returns:
            (total_loss, aux) — `aux` carries metrics and the imagination
            returns used for the post-step `ReturnEMA` update.
        """
        cfg = self.cfg
        B, T = replay_batch_shape(batch)

        rng_key, k_fwd = jax.random.split(rng_key)
        forward = self._world_model_forward(params, batch, k_fwd)

        wm_result = world_model_loss(
            forward=forward,
            params=params,
            batch=batch,
            modules=self._modules,
            cfg=cfg,
            twohot=self.twohot,
        )

        rng_key, k_behavior = jax.random.split(rng_key)
        behavior_result = behavior_loss(
            forward=forward,
            params=params,
            modules=self._modules,
            cfg=cfg,
            twohot=self.twohot,
            slow_critic_params=slow_critic_params,
            ema_state=ema_state,
            return_ema=self.return_ema,
            rng_key=k_behavior,
            B=B,
            T=T,
        )

        rep_result = representation_loss(
            forward=forward,
            batch=batch,
            params=params,
            modules=self._modules,
            cfg=cfg,
            twohot=self.twohot,
            slow_critic_params=slow_critic_params,
            imag_ret=behavior_result.imag_returns,
            B=B,
            T=T,
        )

        losses = {
            **wm_result.losses,
            **behavior_result.losses,
            **rep_result.losses,
        }
        agent_loss = _weighted_total_loss(cfg, losses)
        # The decoder is a stop-gradient visualisation probe. Add its detached
        # reconstruction loss only to the optimiser objective so the decoder
        # learns to read the current latent, while the agent/RSSM/encoder see
        # exactly the same objective as decoder-free runs.
        total_loss = agent_loss
        if cfg.decoder:
            total_loss = total_loss + cfg.scale_decoder * losses["decoder"]

        # ---- Metrics ----
        metrics = {
            **wm_result.metrics,
            **behavior_result.metrics,
            **rep_result.metrics,
        }
        _add_loss_metrics(metrics, losses)
        _add_encoder_l2_metric(metrics, params)

        # ---- Hybrid contribution diagnostics (3D-50) ----
        # Re-split the fused embed into its CNN and gated-VGGT branches via the
        # encoder's `branches` method (shares params with the forward pass) and
        # log how much each modality drives the latent. `gate` starts at 0 and
        # opens over training; `*_frac` is each branch's share of the embed norm.
        if cfg.encoder_type in ("hybrid", "vggt_house_context"):
            _add_hybrid_contribution_metrics(
                metrics,
                cfg=cfg,
                params=params,
                forward=forward,
                B=B,
                T=T,
            )

        aux = AgentLossAux(
            metrics=metrics,
            imag_returns=behavior_result.imag_returns.reshape(-1),
            agent_loss=agent_loss,
        )
        return total_loss, aux
