"""DreamerV3 configuration."""

import dataclasses
from pathlib import Path

import yaml


@dataclasses.dataclass
class DreamerConfig:
    # RSSM
    recurrent_size: int = 512
    latent_length: int = 32
    latent_classes: int = 32
    hidden_size: int = 512
    num_layers: int = 2

    # CNN encoder/decoder
    cnn_depth: int = 48
    kernel_size: int = 4
    stride: int = 2

    # Training
    batch_size: int = 16
    sequence_length: int = 50
    learning_rate: float = 1e-4
    max_grad_norm: float = 100.0
    total_steps: int = 1_000_000
    prefill_steps: int = 5_000
    train_every: int = 5
    seed: int = 0

    # Imagination
    imagination_horizon: int = 15
    gamma: float = 0.997
    lambda_: float = 0.95
    entropy_scale: float = 3e-4

    # Losses
    kl_free_nats: float = 1.0
    kl_balance: float = 0.8

    # Replay
    buffer_capacity: int = 1_000_000

    # Environment
    action_size: int = 4  # FORWARD, LEFT, RIGHT, STOP
    obs_shape: tuple = (4, 256, 256)  # RGB(3) + depth(1), channels-first

    # Logging
    wandb_project: str = "3d-vla-objectnav"
    log_interval: int = 100
    checkpoint_dir: str = "checkpoints"

    @property
    def latent_size(self) -> int:
        return self.latent_length * self.latent_classes

    @property
    def state_size(self) -> int:
        return self.recurrent_size + self.latent_size

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DreamerConfig":
        with open(path) as f:
            overrides = yaml.safe_load(f) or {}
        # Convert list to tuple for obs_shape
        if "obs_shape" in overrides:
            overrides["obs_shape"] = tuple(overrides["obs_shape"])
        return cls(**overrides)
