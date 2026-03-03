from enum import IntEnum

import torch.nn as nn
from torch import Tensor


class HabitatAction(IntEnum):
    STOP = 0
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    LOOK_UP = 4
    LOOK_DOWN = 5


class DiscreteActionHead(nn.Module):
    def __init__(self, cond_dim: int = 256, n_actions: int = 6, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, cond: Tensor) -> Tensor:
        return self.net(cond)

    def sample(self, cond: Tensor) -> Tensor:
        return self.forward(cond).argmax(dim=-1)
