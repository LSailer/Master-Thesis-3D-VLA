from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class R2DreamerConfig:
    # --- Environment ---
    obs_shape: Tuple[int, ...] = (64, 64, 3)
    num_actions: int = 18
    max_episode_steps: int = 27000

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
    encoder_depth: int = 16
    encoder_kernel: int = 5
    encoder_mults: Tuple[int, ...] = (2, 3, 4, 4)

    # --- MLP heads ---
    mlp_units: int = 256
    reward_layers: int = 1
    cont_layers: int = 1
    actor_layers: int = 3
    critic_layers: int = 3
    twohot_bins: int = 255

    # --- Projector (Barlow Twins) ---
    barlow_lambda: float = 5e-4

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
    act_entropy: float = 3e-4
    unimix_ratio: float = 0.01
    slow_target_fraction: float = 0.02

    # --- Replay ---
    buffer_capacity: int = 500_000
    prefill_steps: int = 5000

    # --- Run ---
    total_steps: int = 1_000_000
    log_every: int = 250
    save_every: int = 50_000
    seed: int = 0
    logdir: str = "output/r2dreamer"

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
