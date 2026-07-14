"""Agent-owned R2Dreamer model, optimizer, and loss configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# Model-size presets from the R2-Dreamer table. Each row scales the RSSM width,
# stochastic latent shape, CNN encoder depth, and prediction-head MLP width.
LATENT_PRESETS: dict[str, dict[str, int]] = {
    "12m": {
        "deter_size": 2048,
        "hidden_size": 256,
        "stoch_classes": 32,
        "stoch_discrete": 16,
        "encoder_depth": 16,
        "mlp_units": 256,
    },
    "25m": {
        "deter_size": 3072,
        "hidden_size": 384,
        "stoch_classes": 32,
        "stoch_discrete": 24,
        "encoder_depth": 24,
        "mlp_units": 384,
    },
    "50m": {
        "deter_size": 4096,
        "hidden_size": 512,
        "stoch_classes": 32,
        "stoch_discrete": 32,
        "encoder_depth": 32,
        "mlp_units": 512,
    },
    "100m": {
        "deter_size": 6144,
        "hidden_size": 768,
        "stoch_classes": 32,
        "stoch_discrete": 48,
        "encoder_depth": 48,
        "mlp_units": 768,
    },
    "200m": {
        "deter_size": 8192,
        "hidden_size": 1024,
        "stoch_classes": 32,
        "stoch_discrete": 64,
        "encoder_depth": 64,
        "mlp_units": 1024,
    },
    "400m": {
        "deter_size": 12288,
        "hidden_size": 1536,
        "stoch_classes": 32,
        "stoch_discrete": 96,
        "encoder_depth": 96,
        "mlp_units": 1536,
    },
}


@dataclass
class R2DreamerConfig:
    """Configuration consumed by ``R2DreamerAgent``.

    Trainer-loop ownership lives in ``TrainerConfig``. The current
    ``R2DreamerAgent`` and trainer still read interface and sampling fields from
    this dataclass, so those fields remain here until that constructor boundary
    is split.
    """

    # --- Environment / agent interface ---
    obs_shape: tuple[int, ...] | Mapping[str, tuple[int, ...]] = (3, 64, 64)
    num_actions: int = 4
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
    encoder_type: str = "cnn"
    encoder_module_cls: Any = None
    encoder_depth: int = 16
    encoder_kernel: int = 5
    encoder_mults: tuple[int, ...] = (2, 3, 4, 4)
    vggt_feature_dim: int = 4116
    vggt_embed_dim: int = 1024
    vggt_mlp_layers: int = 1
    vggt_token_transformer_layers: int = 2
    vggt_token_transformer_heads: int = 8
    vggt_token_projection_dim: int = 256
    vggt_token_transformer_mlp_ratio: int = 2
    vggt_token_transformer_dropout: float = 0.0
    vggt_keep_register_tokens: bool = True
    vggt_token_count: int = 1374
    vggt_token_dim: int = 1024
    vggt_house_global_rgb_branch: bool = False
    mlp_vggt_hidden: int = 1024
    mlp_vggt_layers: int = 2
    # House-branch metric XYZ normalization for the MLP/Hybrid house-points
    # encoders: "symlog" compresses channels [:3] (default), "none" passes raw
    # coordinates. PointNet/GNN house encoders ignore this knob.
    house_point_norm: str = "symlog"
    decoder: bool = False
    scale_decoder: float = 1.0
    design_notes: str = ""
    encoder_input_contract: dict[str, Any] | None = None
    compute_dtype: str = "bfloat16"
    # Extend compute_dtype from the token transformer + replay scalars to the
    # whole JAX model (encoders, RSSM, heads) — mixed precision with float32
    # master params and float32-pinned logits. Off by default because
    # compute_dtype was historically a near no-op for CNN/house encoders and
    # existing runs compare against that behavior.
    full_bf16: bool = False

    # --- MLP heads ---
    mlp_units: int = 256
    mlp_layers_reward: int = 1
    mlp_layers_cont: int = 1
    mlp_layers_actor: int = 3
    mlp_layers_critic: int = 3
    twohot_bins: int = 255

    # --- Projector (Barlow Twins) ---
    barlow_lambda: float = 5e-4
    barlow_stop_grad: bool = True

    # --- Training batch cadence ---
    batch_size: int = 16
    seq_len: int = 64
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

    # --- Behavior / value targets ---
    imagination_horizon: int = 15
    horizon: int = 333
    lamb: float = 0.95
    kl_free: float = 1.0
    act_entropy: float = 3e-2
    unimix_ratio: float = 0.01
    slow_target_fraction: float = 0.02

    # --- Reward shaping ---
    step_penalty: float = -0.01
    success_bonus: float = 10.0

    # --- Replay / run metadata ---
    buffer_capacity: int = 500_000
    prefill_steps: int = 5000
    total_steps: int = 1_000_000
    log_every: int = 250
    save_every: int = 50_000
    seed: int = 0
    logdir: str = "output/runs/r2dreamer"

    @property
    def stoch_size(self) -> int:
        """Return flattened stochastic latent size."""
        return self.stoch_classes * self.stoch_discrete

    @property
    def feat_size(self) -> int:
        """Return RSSM feature size consumed by prediction heads."""
        return self.stoch_size + self.deter_size

    @classmethod
    def size_25m(cls) -> "R2DreamerConfig":
        """Return the 25M-parameter table preset."""
        return cls(**LATENT_PRESETS["25m"])
