import math
from typing import Tuple

import torch
from torch import nn


class NavigationGraphEncoder(nn.Module):
    """Masked attention encoder for the relative navigation observation.

    Expected input layout::

        [goal_rel_x, goal_rel_y, rvx, rvy, rtheta, robot_radius,
         (rel_x, rel_y, hvx, hvy, human_radius, mask) * K]

    The number of *valid* humans can vary from sample to sample. Unused slots
    are ignored through the existing mask and the encoder always returns a
    fixed-dimensional latent representation.
    """

    ROBOT_GOAL_DIM = 6
    HUMAN_BLOCK_DIM = 6
    HUMAN_FEATURE_DIM = 5

    def __init__(
        self,
        human_hidden_dim: int = 64,
        robot_goal_hidden_dim: int = 64,
        latent_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.human_hidden_dim = int(human_hidden_dim)
        self.robot_goal_hidden_dim = int(robot_goal_hidden_dim)
        self.latent_dim = int(latent_dim)

        self.robot_goal_encoder = nn.Sequential(
            nn.Linear(self.ROBOT_GOAL_DIM, self.robot_goal_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.robot_goal_hidden_dim, self.robot_goal_hidden_dim),
            nn.ReLU(),
        )
        self.human_encoder = nn.Sequential(
            nn.Linear(self.HUMAN_FEATURE_DIM, self.human_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.human_hidden_dim, self.human_hidden_dim),
            nn.ReLU(),
        )

        # Robot/goal context queries the surrounding-human set.
        self.query = nn.Linear(
            self.robot_goal_hidden_dim, self.human_hidden_dim, bias=False
        )
        self.key = nn.Linear(
            self.human_hidden_dim, self.human_hidden_dim, bias=False
        )
        self.value = nn.Linear(
            self.human_hidden_dim, self.human_hidden_dim, bias=False
        )

        self.fusion = nn.Sequential(
            nn.Linear(
                self.robot_goal_hidden_dim + self.human_hidden_dim,
                self.latent_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
        )

    @staticmethod
    def _ensure_batch(obs: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        squeeze = False
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            squeeze = True
        if obs.dim() != 2:
            raise ValueError(
                f"Expected (D,) or (B,D) observation, got {tuple(obs.shape)}"
            )
        return obs, squeeze

    def split_observation(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split observation into robot/goal, human features and validity mask.

        Returns:
            robot_goal: ``(B, 6)``
            human_features: ``(B, K, 5)``
            human_mask: ``(B, K)`` boolean
        """
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32)

        obs, _ = self._ensure_batch(obs)
        width = obs.size(1)
        if width < self.ROBOT_GOAL_DIM:
            raise ValueError(f"Observation width must be >= 6, got {width}")

        tail = width - self.ROBOT_GOAL_DIM
        if tail % self.HUMAN_BLOCK_DIM != 0:
            raise ValueError(
                "Expected relative observation width 6 + 6*K; "
                f"received {width}"
            )

        robot_goal = obs[:, : self.ROBOT_GOAL_DIM]
        k = tail // self.HUMAN_BLOCK_DIM
        if k == 0:
            human_features = obs.new_zeros((obs.size(0), 0, self.HUMAN_FEATURE_DIM))
            human_mask = torch.zeros(
                (obs.size(0), 0), dtype=torch.bool, device=obs.device
            )
            return robot_goal, human_features, human_mask

        blocks = obs[:, self.ROBOT_GOAL_DIM :].reshape(
            obs.size(0), k, self.HUMAN_BLOCK_DIM
        )
        human_features = blocks[:, :, : self.HUMAN_FEATURE_DIM]
        human_mask = blocks[:, :, 5] > 0.5
        return robot_goal, human_features, human_mask

    def forward(self, obs: torch.Tensor, return_attention: bool = False):
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32)

        obs, squeeze = self._ensure_batch(obs)
        robot_goal, humans, human_mask = self.split_observation(obs)
        robot_goal_h = self.robot_goal_encoder(robot_goal)

        if humans.size(1) == 0:
            crowd_h = robot_goal_h.new_zeros(
                (robot_goal_h.size(0), self.human_hidden_dim)
            )
            attention = robot_goal_h.new_zeros((robot_goal_h.size(0), 0))
        else:
            human_h = self.human_encoder(humans)
            q = self.query(robot_goal_h).unsqueeze(1)
            k = self.key(human_h)
            v = self.value(human_h)

            scores = (q * k).sum(dim=-1) / math.sqrt(self.human_hidden_dim)
            scores = scores.masked_fill(~human_mask, -1e9)

            attention = torch.softmax(scores, dim=1)
            attention = attention * human_mask.to(attention.dtype)

            # Softmax over an all-masked row is not meaningful. Explicitly map
            # that case to a zero crowd context and re-normalize valid rows.
            has_valid = human_mask.any(dim=1, keepdim=True)
            denom = attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
            attention = torch.where(
                has_valid,
                attention / denom,
                torch.zeros_like(attention),
            )
            crowd_h = torch.sum(attention.unsqueeze(-1) * v, dim=1)

        latent = self.fusion(torch.cat([robot_goal_h, crowd_h], dim=-1))

        if squeeze:
            latent = latent.squeeze(0)
            attention = attention.squeeze(0)

        if return_attention:
            return latent, attention
        return latent
