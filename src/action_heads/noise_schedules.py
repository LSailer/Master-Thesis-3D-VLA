import math

import torch
from torch import Tensor


# Linear beta schedule from DDPM.
def linear_beta_schedule(n_steps: int) -> Tensor:
    return torch.linspace(1e-4, 0.02, n_steps)


# Cosine beta schedule (Nichol & Dhariwal 2021).
def cosine_beta_schedule(n_steps: int, s: float = 0.008) -> Tensor:
    steps = torch.arange(n_steps + 1) / n_steps
    alpha_bar = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return torch.clamp(betas, max=0.999)


# Cumulative product of (1 - betas).
def compute_alpha_bar(betas: Tensor) -> Tensor:
    return torch.cumprod(1 - betas, dim=0)
