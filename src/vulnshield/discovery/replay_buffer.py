"""Prioritised Experience Replay Buffer for TD3 Training."""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple
import torch
import numpy as np


class Transition(NamedTuple):
    state: torch.Tensor         # (obs_dim,)
    action: torch.Tensor        # (action_dim,)
    reward: float
    next_state: torch.Tensor    # (obs_dim,)
    done: bool


class ReplayBuffer:
    """Fixed-size circular experience replay buffer for TD3.

    Stores transitions (s, a, r, s', done) and supports uniform
    random mini-batch sampling for off-policy training.

    Args:
        obs_dim: Dimension of the observation space.
        action_dim: Dimension of the action space.
        capacity: Maximum number of transitions to store.
        device: Device for returned batch tensors.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        capacity: int = 10_000,
        device: Optional[torch.device] = None
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.device = device or torch.device("cpu")

        self._states = torch.zeros((capacity, obs_dim), dtype=torch.float32)
        self._actions = torch.zeros((capacity, action_dim), dtype=torch.float32)
        self._rewards = torch.zeros((capacity, 1), dtype=torch.float32)
        self._next_states = torch.zeros((capacity, obs_dim), dtype=torch.float32)
        self._dones = torch.zeros((capacity, 1), dtype=torch.float32)

        self._ptr = 0
        self._size = 0

    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool
    ) -> None:
        """Store a single transition in the buffer."""
        idx = self._ptr
        self._states[idx] = state.cpu()
        self._actions[idx] = action.cpu()
        self._rewards[idx, 0] = reward
        self._next_states[idx] = next_state.cpu()
        self._dones[idx, 0] = float(done)

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a random mini-batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) tensors.

        Raises:
            RuntimeError: If buffer has fewer transitions than batch_size.
        """
        if self._size < batch_size:
            raise RuntimeError(
                f"Buffer contains {self._size} transitions, "
                f"need at least {batch_size} for sampling."
            )
        indices = np.random.randint(0, self._size, size=batch_size)
        dev = self.device
        return (
            self._states[indices].to(dev),
            self._actions[indices].to(dev),
            self._rewards[indices].to(dev),
            self._next_states[indices].to(dev),
            self._dones[indices].to(dev)
        )

    def __len__(self) -> int:
        return self._size

    @property
    def is_ready(self) -> bool:
        return self._size > 0
