"""DreamerV3 configuration dataclass."""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class DreamerConfig:
    obs_dim: int = 4
    hidden: int = 8
    obs_shape: Tuple[int, ...] = (64, 64, 3)
    num_actions: int = 6
    batch_size: int = 2
    seq_len: int = 4
    eval_every: int = 100_000
    eval_episodes: int = 10
