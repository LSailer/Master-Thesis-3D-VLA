"""DreamerV3 configuration — flat dataclass with sensible defaults."""

import dataclasses


@dataclasses.dataclass
class DreamerConfig:
    # Environment
    obs_shape: tuple = (3, 256, 256)  # C, H, W
    num_actions: int = 4  # STOP, FORWARD, LEFT, RIGHT
    max_episode_steps: int = 500
    reward_type: str = "geodesic_delta"  # or "sparse"
    split: str = "train"  # dataset split: train, val, val_mini

    # RSSM
    hidden_size: int = 512  # GRU hidden
    latent_classes: int = 32  # categorical latent classes
    latent_dims: int = 32  # categorical latent dims

    @property
    def stoch_size(self) -> int:
        return self.latent_classes * self.latent_dims

    # Networks
    encoder_depth: int = 48  # CNN channel multiplier
    mlp_hidden: int = 512
    mlp_layers: int = 2

    # Training
    batch_size: int = 16
    seq_len: int = 64
    imagination_horizon: int = 15
    lr_world: float = 1e-4
    lr_actor: float = 3e-5
    lr_critic: float = 3e-5
    max_grad_norm: float = 100.0
    discount: float = 0.997
    lambda_: float = 0.95
    free_nats: float = 1.0
    kl_weight: float = 1.0
    entropy_scale: float = 3e-4

    # Replay
    buffer_capacity: int = 1_000_000
    prefill_steps: int = 5000

    # Run
    total_steps: int = 5_000_000
    log_every: int = 1000
    save_every: int = 50_000
    seed: int = 0
    logdir: str = "output/dreamerv3"
