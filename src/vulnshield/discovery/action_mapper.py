"""Continuous-to-Discrete Action Mapper for TD3 Channel Discovery.

The TD3 actor outputs a continuous action vector in [-1, 1]^d.
This module maps that vector to a valid discrete (layer_idx, channel_idx) pair
that the FaultInjector can act on.

Mapping strategy:
  - action[0]  → layer index   (linearly scaled to [0, num_layers-1])
  - action[1]  → channel index (linearly scaled to [0, out_channels-1] for that layer)
"""

from __future__ import annotations

from typing import List, Tuple
import torch


class ActionMapper:
    """Maps continuous TD3 actor output to discrete (layer_idx, channel_idx).

    Args:
        layer_channel_counts: Ordered list of (layer_name, num_channels) for
            all injectable Conv2d layers, in model execution order.
    """

    def __init__(self, layer_channel_counts: List[Tuple[str, int]]):
        if not layer_channel_counts:
            raise ValueError("layer_channel_counts must be non-empty.")
        self.layer_channel_counts = layer_channel_counts
        self.num_layers = len(layer_channel_counts)
        # Action dimension: 2 (one per axis)
        self.action_dim = 2

    @property
    def max_channels(self) -> int:
        return max(n for _, n in self.layer_channel_counts)

    def decode(self, action: torch.Tensor) -> Tuple[str, int]:
        """Convert a continuous action vector to a discrete (layer_name, channel_idx).

        Args:
            action: Tensor of shape (2,) with values in [-1, 1].

        Returns:
            Tuple of (layer_name, channel_idx).
        """
        # Clamp to valid range
        a = action.clamp(-1.0, 1.0)

        # Map action[0] → layer_idx in [0, num_layers-1]
        layer_idx = int(((a[0].item() + 1.0) / 2.0) * (self.num_layers - 1 + 1e-6))
        layer_idx = min(max(layer_idx, 0), self.num_layers - 1)

        layer_name, num_channels = self.layer_channel_counts[layer_idx]

        # Map action[1] → channel_idx in [0, num_channels-1]
        channel_idx = int(((a[1].item() + 1.0) / 2.0) * (num_channels - 1 + 1e-6))
        channel_idx = min(max(channel_idx, 0), num_channels - 1)

        return layer_name, channel_idx

    def encode(self, layer_idx: int, channel_idx: int) -> torch.Tensor:
        """Encode a discrete (layer_idx, channel_idx) pair back to a continuous action.

        Useful for seeding the replay buffer with known high-value actions.

        Args:
            layer_idx: Index into layer_channel_counts.
            channel_idx: Channel index within that layer.

        Returns:
            Tensor of shape (2,) with values in [-1, 1].
        """
        _, num_channels = self.layer_channel_counts[layer_idx]
        a0 = (2.0 * layer_idx / max(self.num_layers - 1, 1)) - 1.0
        a1 = (2.0 * channel_idx / max(num_channels - 1, 1)) - 1.0
        return torch.tensor([a0, a1], dtype=torch.float32)
