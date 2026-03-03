import math

import torch
import torch.nn as nn


# Sinusoidal timestep embeddings.
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


# MLP-based noise prediction network for diffusion action heads.
class MLPDenoiser(nn.Module):
    def __init__(
        self,
        action_dim: int,
        cond_dim: int,
        chunk_size: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.time_emb = SinusoidalPosEmb(hidden_dim)
        self.input_proj = nn.Linear(
            action_dim * chunk_size + hidden_dim + cond_dim, hidden_dim
        )
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
                for _ in range(n_layers)
            ]
        )
        self.out = nn.Linear(hidden_dim, action_dim * chunk_size)

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        t_emb = self.time_emb(t)
        h = torch.cat([x_t.flatten(1), t_emb, cond], dim=-1)
        h = self.input_proj(h)
        for layer in self.layers:
            h = layer(h) + h
        return self.out(h).unflatten(-1, (self.chunk_size, self.action_dim))


# Exponential moving average for model weights.
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.shadow = {k: v.clone() for k, v in model.state_dict().items()}
        self.decay = decay

    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v

    def apply(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow)
