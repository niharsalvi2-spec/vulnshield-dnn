"""TD3 Actor Network — Deterministic Policy MLP.

Architecture:
    Linear(obs_dim → 256) → LayerNorm → ReLU
    Linear(256 → 256) → LayerNorm → ReLU
    Linear(256 → action_dim) → Tanh          [output ∈ (-1, 1)]

The output is clipped to (-1, 1) via Tanh, which the ActionMapper then
scales to valid (layer_idx, channel_idx) pairs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TD3Actor(nn.Module):
    """Deterministic policy network for TD3 fault channel discovery.

    Args:
        obs_dim: Dimension of the state observation vector.
        action_dim: Dimension of the continuous action space.
        hidden_dim: Width of the hidden layers.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Compute deterministic action from state.

        Args:
            state: Observation tensor (B, obs_dim) or (obs_dim,).

        Returns:
            Action tensor in (-1, 1)^action_dim.
        """
        return self.net(state)
