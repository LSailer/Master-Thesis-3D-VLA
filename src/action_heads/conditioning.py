import torch
import torch.nn as nn
from torch import Tensor


class FusionModule(nn.Module):
    def __init__(
        self,
        clip_dim: int = 768,
        spatial_channels: int = 5,
        pose_dim: int = 9,
        cond_dim: int = 256,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(spatial_channels, 32, 3, 1, 1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.pose_encoder = nn.Linear(pose_dim, hidden_dim)
        self.goal_encoder = nn.Linear(clip_dim, hidden_dim)
        self.fuse = nn.Sequential(
            nn.Linear(32 + hidden_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, cond_dim),
        )

    def forward(
        self,
        world_points: Tensor,
        depth: Tensor,
        relevance: Tensor,
        pose_enc: Tensor,
        goal_emb: Tensor,
    ) -> Tensor:
        spatial = torch.cat([world_points, depth, relevance], dim=1)
        spatial_feat = self.spatial_encoder(spatial)
        pose_feat = self.pose_encoder(pose_enc)
        goal_feat = self.goal_encoder(goal_emb)
        return self.fuse(torch.cat([spatial_feat, pose_feat, goal_feat], dim=-1))
