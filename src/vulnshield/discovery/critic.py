"""TD3 Twin Critic Networks — Two Independent Q-Function MLPs.

TD3 uses two independent Q-networks to mitigate overestimation bias:
    Q1(s, a) and Q2(s, a)

During updates, the minimum of Q1 and Q2 is used for the target:
    y = r + γ * min(Q1(s', ã), Q2(s', ã))

where ã = π_target(s') + clipped_noise.

Architecture per critic:
    Linear(obs_dim + action_dim → 256) → ReLU
    Linear(256 → 256) → ReLU
    Linear(256 → 1)
"""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn


class _SingleCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class TD3TwinCritic(nn.Module):
    """Twin Q-function critics for TD3.

    Args:
        obs_dim: Dimension of the state observation vector.
        action_dim: Dimension of the continuous action space.
        hidden_dim: Width of each critic's hidden layers.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.q1 = _SingleCritic(obs_dim, action_dim, hidden_dim)
        self.q2 = _SingleCritic(obs_dim, action_dim, hidden_dim)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute both Q-values for (state, action).

        Returns:
            Tuple of (Q1_value, Q2_value), each of shape (B, 1).
        """
        return self.q1(state, action), self.q2(state, action)

    def q1_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute only Q1 — used for actor update."""
        return self.q1(state, action)
