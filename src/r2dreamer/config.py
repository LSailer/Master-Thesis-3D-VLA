from dataclasses import dataclass, field
from typing import Any, Tuple


# Latent-size ablation presets (3D-50). Each maps a `--latent_preset` name to the
# RSSM-size field overrides it applies; explicit CLI flags still win over a preset.
# Table-driven so adding a preset is a single entry, not another if/elif branch.
LATENT_PRESETS: dict[str, dict[str, int]] = {
    "small": {"deter_size": 1024, "stoch_classes": 24, "stoch_discrete": 12},
    "large": {"deter_size": 4096, "stoch_classes": 48, "stoch_discrete": 24},
}


@dataclass
class R2DreamerConfig:
    # --- Environment ---
    obs_shape: Tuple[int, ...] = (3, 64, 64)  # CHW format (matches JAX codebase)
    num_actions: int = 4   # Habitat default (STOP, FORWARD, TURN_LEFT, TURN_RIGHT); Crafter overrides to 17 in train.py
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
    encoder_type: str = "cnn"  # canonical set: encoder_registry in src.r2dreamer.launch.registries
    encoder_module_cls: Any = None  # Flax nn.Module class; sourced from EncoderSpec.module_cls
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
    mlp_vggt_hidden: int = 1024   # hidden width of the hybrid VGGT-branch MLP
    mlp_vggt_layers: int = 2      # depth of the hybrid VGGT-branch MLP
    # --- Debug decoder probe (image reconstruction; OFF by default; 3D-51) ---
    # When enabled, trains a ConvDecoder from stop-gradient RSSM features for
    # W&B visualisation. It does not backprop reconstruction loss into the
    # encoder/RSSM/actor-critic objective.
    decoder: bool = False         # build a ConvDecoder visualisation probe
    scale_decoder: float = 1.0    # weight of decoder-only reconstruction loss
    design_notes: str = ""

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
