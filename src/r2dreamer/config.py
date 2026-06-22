from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Tuple


# Latent-size ablation presets (3D-50). Each maps a `--latent_preset` name to the
# RSSM-size field overrides it applies; explicit CLI flags still win over a preset.
# Table-driven so adding a preset is a single entry, not another if/elif branch.
LATENT_PRESETS: dict[str, dict[str, int]] = {
    "small": {"deter_size": 1024, "stoch_classes": 24, "stoch_discrete": 12},
    "large": {"deter_size": 4096, "stoch_classes": 48, "stoch_discrete": 24},
}

ReplayComponent = Literal["image", "world_points", "wp_cp", "tokens", "features"]
_ALLOWED_REPLAY_COMPONENTS = {"image", "world_points", "wp_cp", "tokens", "features"}


@dataclass(frozen=True)
class ObservationDims:
    """Dimension knobs for observation/replay layouts.

    Store knobs here, derive shapes from properties. This avoids independent
    config fields like ``wp_side=37`` and ``vggt_feature_dim=4116`` drifting.
    """

    render_size: int = 518
    replay_image_size: int = 64
    wp_side: int = 37
    camera_pose_dim: int = 9
    xyz_channels: int = 3
    token_count: int = 1374
    token_dim: int = 1024

    @property
    def render_shape(self) -> tuple[int, int, int]:
        return (3, self.render_size, self.render_size)

    @property
    def image_shape(self) -> tuple[int, int, int]:
        return (3, self.replay_image_size, self.replay_image_size)

    @property
    def world_points_shape(self) -> tuple[int, int, int]:
        return (self.xyz_channels, self.wp_side, self.wp_side)

    @property
    def camera_pose_shape(self) -> tuple[int]:
        return (self.camera_pose_dim,)

    @property
    def wp_cp_dim(self) -> int:
        return self.xyz_channels * self.wp_side * self.wp_side + self.camera_pose_dim

    @property
    def wp_cp_shape(self) -> tuple[int]:
        return (self.wp_cp_dim,)

    @property
    def token_shape(self) -> tuple[int, int]:
        return (self.token_count, self.token_dim)

    @property
    def flat_token_shape(self) -> tuple[int]:
        return (self.token_count * self.token_dim,)


@dataclass(frozen=True)
class ReplayObservationConfig:
    """Replay fields requested by a run.

    ``components`` controls what buffer_obs stores. Single-component configs map
    to the existing array replay path; multi-component configs map to dict replay.
    """

    components: tuple[ReplayComponent, ...] = ("image",)
    image_dtype: str = "uint8"
    feature_dtype: str = "float32"
    normalize_image: bool = True
    feature_shape: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        invalid = set(self.components) - _ALLOWED_REPLAY_COMPONENTS
        if invalid:
            raise ValueError(f"unknown replay components: {sorted(invalid)}")
        if "features" in self.components and self.feature_shape is None:
            raise ValueError("feature_shape is required for generic 'features'")

    @property
    def stores_image(self) -> bool:
        return "image" in self.components

    @property
    def stores_features(self) -> bool:
        return any(
            component in self.components
            for component in ("world_points", "wp_cp", "tokens", "features")
        )


@dataclass(frozen=True)
class ObservationRunConfig:
    """Observation/replay config for one run, independent of model hyperparams."""

    encoder: str = "cnn"
    dims: ObservationDims = field(default_factory=ObservationDims)
    replay: ReplayObservationConfig = field(default_factory=ReplayObservationConfig)

    def replay_field_shapes(self) -> dict[str, tuple[int, ...]]:
        fields: dict[str, tuple[int, ...]] = {}
        dims = self.dims
        if "image" in self.replay.components:
            fields["image"] = dims.image_shape
        if "world_points" in self.replay.components:
            fields["world_points"] = dims.world_points_shape
            fields["camera_pose"] = dims.camera_pose_shape
        if "wp_cp" in self.replay.components:
            fields["wp_cp"] = dims.wp_cp_shape
        if "tokens" in self.replay.components:
            fields["tokens"] = dims.flat_token_shape
        if "features" in self.replay.components:
            fields["features"] = self.replay.feature_shape or ()
        return fields

    def replay_buffer_shape(self) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
        fields = self.replay_field_shapes()
        if len(fields) == 1:
            return next(iter(fields.values()))
        return fields

    def replay_field_dtypes(self) -> dict[str, str]:
        return {
            name: (
                self.replay.image_dtype
                if name == "image"
                else self.replay.feature_dtype
            )
            for name in self.replay_field_shapes()
        }

    def replay_buffer_dtype(self) -> str | dict[str, str]:
        dtypes = self.replay_field_dtypes()
        if len(dtypes) == 1:
            return next(iter(dtypes.values()))
        return dtypes

    def replay_field_normalize(self) -> dict[str, bool]:
        return {
            name: name == "image" and self.replay.normalize_image
            for name in self.replay_field_shapes()
        }

    def replay_buffer_normalize(self) -> bool | dict[str, bool]:
        normalize = self.replay_field_normalize()
        if len(normalize) == 1:
            return next(iter(normalize.values()))
        return normalize


@dataclass
class R2DreamerConfig:
    # --- Environment ---
    obs_shape: Tuple[int, ...] | Mapping[str, Tuple[int, ...]] = (
        3,
        64,
        64,
    )  # CHW format or structured fields
    num_actions: int = 4  # Habitat default (STOP, FORWARD, TURN_LEFT, TURN_RIGHT); Crafter overrides to 17 in train.py
    max_episode_steps: int = 1000

    # --- RSSM ---
    deter_size: int = 2048
    hidden_size: int = 256
    stoch_classes: int = 32
    stoch_discrete: int = 16
    blocks: int = 8
    dyn_layers: int = 1
    obs_layers: int = 1
    img_layers: int = 2

    # --- Encoder ---
    encoder_type: str = (
        "cnn"  # canonical set: encoder_registry in src.r2dreamer.launch.registries
    )
    encoder_module_cls: Any = (
        None  # Flax nn.Module class; sourced from EncoderSpec.module_cls
    )
    encoder_depth: int = 16
    encoder_kernel: int = 5
    encoder_mults: Tuple[int, ...] = (2, 3, 4, 4)
    vggt_feature_dim: int = 4116  # 37*37*3 + 9 (world_points + camera_pose)
    vggt_embed_dim: int = 1024
    # Depth of the VGGT MLP encoders (wp_cp + aggregator), counting hidden
    # Dense->RMSNorm->SiLU blocks before the linear readout. Default 1 keeps the
    # encoder shallow (one hidden block + projection); the experiment runs raise
    # this to 3 to match R2Dreamer's native encoder.mlp.layers (3D-52). Setting
    # 0 collapses wp_cp to the original bare-linear projection.
    vggt_mlp_layers: int = 1
    # Token Transformer over frozen VGGT aggregator tokens (3D-75).
    # Defaults keep the full 1374-token sequence (camera + 4 registers + 37x37
    # patches), project 1024-d tokens down before attention, and return
    # vggt_embed_dim for the existing RSSM observe path.
    vggt_token_transformer_layers: int = 2
    vggt_token_transformer_heads: int = 8
    vggt_token_projection_dim: int = 256
    vggt_token_transformer_mlp_ratio: int = 2
    vggt_token_transformer_dropout: float = 0.0
    vggt_keep_register_tokens: bool = True
    vggt_token_count: int = 1374
    vggt_token_dim: int = 1024
    # --- Hybrid encoder (CNN on RGB + MLP on WP/CP, gated; 3D-50/51/52) ---
    # The hybrid's WP/CP branch has its own width/depth knobs (vggt_mlp_layers
    # above governs the standalone vggt/aggregator encoders).
    mlp_vggt_hidden: int = 1024  # hidden width of the hybrid VGGT-branch MLP
    mlp_vggt_layers: int = 2  # depth of the hybrid VGGT-branch MLP
    # --- Debug decoder probe (image reconstruction; OFF by default; 3D-51) ---
    # When enabled, trains a ConvDecoder from stop-gradient RSSM features for
    # W&B visualisation. It does not backprop reconstruction loss into the
    # encoder/RSSM/actor-critic objective.
    decoder: bool = False  # build a ConvDecoder visualisation probe
    scale_decoder: float = 1.0  # weight of decoder-only reconstruction loss
    design_notes: str = ""
    # JSON-serializable Encoder Input Contract snapshot persisted to durable
    # run metadata. Runtime config may still hold encoder_module_cls; snapshots
    # use stable module names, shapes, dtypes, booleans, and overrides.
    encoder_input_contract: dict[str, Any] | None = None
    # Compute dtype for large encoder activations. Official DreamerV3 uses
    # bfloat16 compute; explicit fp32 probe configs can still override this for
    # comparison hygiene.
    compute_dtype: str = "bfloat16"

    # --- MLP heads ---
    mlp_units: int = 256
    mlp_layers_reward: int = 1
    mlp_layers_cont: int = 1
    mlp_layers_actor: int = 3
    mlp_layers_critic: int = 3
    twohot_bins: int = 255

    # --- Projector (Barlow Twins) ---
    barlow_lambda: float = 5e-4
    # When True (default, matches PyTorch reference), Barlow Twins detaches the
    # encoder side: gradient flows only into the projector + RSSM. Set False
    # for Protocol D — verifies whether the detached encoder is starving the
    # aggregator MLP and VGGT of useful signal.
    barlow_stop_grad: bool = True

    # --- Training ---
    batch_size: int = 16
    seq_len: int = 64
    imagination_horizon: int = 15
    horizon: int = 333
    lamb: float = 0.95
    train_ratio: int = 512

    # --- Optimizer ---
    lr: float = 4e-5
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-20
    warmup_steps: int = 1000

    # --- Adaptive Gradient Clipping (AGC) ---
    agc_clip: float = 0.3
    agc_pmin: float = 1e-3

    # --- Loss scales ---
    scale_barlow: float = 0.05
    scale_dyn: float = 1.0
    scale_rep: float = 0.1
    scale_rew: float = 1.0
    scale_con: float = 1.0
    scale_policy: float = 1.0
    scale_value: float = 1.0
    scale_repval: float = 0.3

    # --- Behavior ---
    kl_free: float = 1.0
    act_entropy: float = 3e-2
    unimix_ratio: float = 0.01
    slow_target_fraction: float = 0.02

    # --- Reward ---
    step_penalty: float = -0.01  # per-step cost (encourages shorter paths)
    success_bonus: float = 10.0  # reward for reaching goal

    # --- Replay ---
    buffer_capacity: int = 500_000
    prefill_steps: int = 5000

    # --- Run ---
    total_steps: int = 1_000_000
    log_every: int = 250
    save_every: int = 50_000
    seed: int = 0
    logdir: str = "output/runs/r2dreamer"

    @property
    def stoch_size(self) -> int:
        return self.stoch_classes * self.stoch_discrete

    @property
    def feat_size(self) -> int:
        return self.stoch_size + self.deter_size

    @classmethod
    def size25M(cls) -> "R2DreamerConfig":
        return cls(
            deter_size=3072,
            hidden_size=384,
            stoch_discrete=24,
            encoder_depth=24,
            mlp_units=384,
        )


@dataclass
class TrainerConfig:
    """Controls the training loop (separate from R2DreamerConfig model arch)."""

    output_dir: str = "output/runs/r2dreamer"
    total_steps: int = 10_000_000
    prefill_steps: int = 5000
    log_every: int = 250
    checkpoint_every: int = 50_000
    seed: int = 0

    # When True, a fully completed run hard-exits the process (os._exit(0))
    # after the final checkpoint, MANIFEST, and W&B are flushed — skipping
    # habitat_sim's GL teardown, which SIGABRTs ("no current context") on some
    # magnum builds and would otherwise poison the exit code of a successful
    # run. Set by the SLURM launcher (env R2DREAMER_HARD_EXIT_ON_FINISH=1);
    # left False for notebook/test callers so they keep the normal close() path
    # and real failures still surface a non-zero exit.
    hard_exit_on_finish: bool = False

    # WandB (None = disabled)
    wandb_project: str | None = "3d-vla-objectnav"
    wandb_name: str | None = None
    wandb_tags: list[str] = field(default_factory=lambda: ["r2dreamer"])
    # Resume an existing W&B run (e.g. "87u0l6dy"). Requires the run to exist.
    wandb_id: str | None = None
    video_log_every: int = 0
    video_log_episodes: int = 0

    # Deterministic Val-Episode-Loop. val_every=0 disables. Requires a
    # val_env to be passed to Trainer. Default off keeps production runs
    # scalars-only unless validation is explicitly requested.
    val_every: int = 0
    val_episodes: int = 50
    val_video_episodes: int = 0
    val_max_episode_steps: int = 500

    # Resume from checkpoint (.pkl produced by save_checkpoint). When set,
    # restores agent.{params, opt_state, slow_critic_params, ema_state} and
    # offsets the train loop to start at the checkpoint's step.
    resume_from: str | None = None

    # --- Karpathy step-3 diagnostic: overfit a single sampled batch ---
    # When True, the run does the normal prefill, then samples one batch
    # (overfit_batch_size, overfit_seq_len) once, freezes it, and runs
    # agent.train_step on that same batch for overfit_steps iterations.
    # No env rollouts, no validation, no checkpointing.
    overfit_one_batch: bool = False
    overfit_steps: int = 1000
    overfit_batch_size: int = 1
    overfit_seq_len: int = 8
    overfit_min_loss_drop: float = 0.20
