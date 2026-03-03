import torch
import torch.nn as nn
from torch import Tensor

from .common import MLPDenoiser
from .noise_schedules import compute_alpha_bar, cosine_beta_schedule


class DDPMActionHead(nn.Module):
    def __init__(
        self,
        action_dim: int = 4,
        cond_dim: int = 256,
        chunk_size: int = 4,
        n_steps: int = 100,
        hidden_dim: int = 256,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.n_steps = n_steps
        self.denoiser = MLPDenoiser(
            action_dim, cond_dim, chunk_size, hidden_dim, n_layers
        )

        betas = cosine_beta_schedule(n_steps)
        alphas = 1.0 - betas
        alpha_bar = compute_alpha_bar(betas)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

    def forward(self, cond: Tensor, noisy_actions: Tensor, timesteps: Tensor) -> Tensor:
        return self.denoiser(noisy_actions, timesteps.float(), cond)

    @torch.no_grad()
    def sample(self, cond: Tensor, n_samples: int = 1) -> Tensor:
        B = cond.shape[0]
        x = torch.randn(B, self.chunk_size, self.action_dim, device=cond.device)

        for t in reversed(range(self.n_steps)):
            t_batch = torch.full((B,), t, device=cond.device, dtype=torch.long)
            eps_pred = self.forward(cond, x, t_batch)

            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bar[t]
            beta_t = self.betas[t]

            x = (1.0 / alpha_t.sqrt()) * (
                x - (beta_t / (1.0 - alpha_bar_t).sqrt()) * eps_pred
            )

            if t > 0:
                noise = torch.randn_like(x)
                x = x + beta_t.sqrt() * noise

        return x
