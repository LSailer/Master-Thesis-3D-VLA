import torch
import torch.nn as nn
from torch import Tensor

from .common import MLPDenoiser


class FlowMatchingActionHead(nn.Module):
    def __init__(
        self,
        action_dim: int = 4,
        cond_dim: int = 256,
        chunk_size: int = 4,
        hidden_dim: int = 256,
        n_layers: int = 4,
        sigma_min: float = 1e-4,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.sigma_min = sigma_min
        self.denoiser = MLPDenoiser(
            action_dim, cond_dim, chunk_size, hidden_dim, n_layers
        )

    def forward(self, cond: Tensor, x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
        t_expand = t[:, None, None]
        x_t = (1 - t_expand) * x_0 + t_expand * x_1
        return self.denoiser(x_t, t, cond)

    @torch.no_grad()
    def sample(self, cond: Tensor, n_samples: int = 1, n_steps: int = 20) -> Tensor:
        B = cond.shape[0]
        dt = 1.0 / n_steps
        x = torch.randn(B, self.chunk_size, self.action_dim, device=cond.device)
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=cond.device)
            v = self.denoiser(x, t, cond)
            x = x + v * dt
        return x
