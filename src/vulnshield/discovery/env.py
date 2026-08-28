"""Fault Discovery RL Environment.

Wraps FaultInjector + evaluate_model into a minimal Gym-style RL environment.

State (observation) vector:
    [layer_fraction, channel_fraction, clean_acc_norm, last_delta_norm]

    - layer_fraction    = layer_idx / (num_layers - 1)         in [0, 1]
    - channel_fraction  = channel_idx / (max_channels - 1)     in [0, 1]
    - clean_acc_norm    = clean_accuracy / 100.0               in [0, 1]
    - last_delta_norm   = last ΔA / 100.0                      in [-1, 1] clipped

Reward:
    r = ΔA(l, c) = clean_accuracy − fault_accuracy   (higher → better)

Episode terminates after `budget` steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vulnshield.fault_injection.fault_injector import FaultInjector
from vulnshield.training.evaluator import evaluate_model
from vulnshield.discovery.action_mapper import ActionMapper


OBS_DIM = 4    # State vector dimensionality


@dataclass
class StepResult:
    observation: torch.Tensor   # shape (OBS_DIM,)
    reward: float
    done: bool
    info: dict


class FaultDiscoveryEnv:
    """RL environment for discovering vulnerable channels via fault injection.

    Args:
        model: Pre-trained neural network (frozen weights).
        dataloader: Evaluation DataLoader (eval_fault_loader, ~1000 samples).
        clean_accuracy: Pre-computed clean accuracy (%).
        budget: Max fault evaluation steps per episode.
        device: Compute device.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        clean_accuracy: float,
        budget: int = 50,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.dataloader = dataloader
        self.clean_accuracy = clean_accuracy
        self.budget = budget
        self.device = device or torch.device("cpu")

        self.injector = FaultInjector(model)
        self.layer_channel_counts = self.injector.list_injectable_layers()
        self.action_mapper = ActionMapper(self.layer_channel_counts)

        self._step_count = 0
        self._last_delta = 0.0
        self._last_layer_idx = 0
        self._last_channel_idx = 0

    @property
    def obs_dim(self) -> int:
        return OBS_DIM

    @property
    def action_dim(self) -> int:
        return self.action_mapper.action_dim

    def reset(self) -> torch.Tensor:
        """Reset environment for a new episode."""
        self._step_count = 0
        self._last_delta = 0.0
        self._last_layer_idx = 0
        self._last_channel_idx = 0
        return self._make_obs()

    def step(self, action: torch.Tensor) -> StepResult:
        """Apply one fault action and return the resulting step.

        Args:
            action: Continuous action tensor of shape (2,).

        Returns:
            StepResult with next observation, reward, done flag, and info dict.
        """
        layer_name, channel_idx = self.action_mapper.decode(action)
        layer_idx = next(
            i for i, (n, _) in enumerate(self.layer_channel_counts) if n == layer_name
        )

        # Inject fault and evaluate
        with self.injector.inject([(layer_name, channel_idx)]):
            result = evaluate_model(self.model, self.dataloader, device=self.device)

        delta = self.clean_accuracy - result.accuracy   # ΔA = reward signal

        self._last_delta = delta
        self._last_layer_idx = layer_idx
        self._last_channel_idx = channel_idx
        self._step_count += 1

        done = self._step_count >= self.budget
        obs = self._make_obs()

        return StepResult(
            observation=obs,
            reward=float(delta),
            done=done,
            info={
                "layer_name": layer_name,
                "channel_idx": channel_idx,
                "fault_accuracy": result.accuracy,
                "delta_accuracy": delta,
                "step": self._step_count
            }
        )

    def _make_obs(self) -> torch.Tensor:
        num_layers = len(self.layer_channel_counts)
        max_channels = self.action_mapper.max_channels

        layer_frac = self._last_layer_idx / max(num_layers - 1, 1)
        _, layer_channels = self.layer_channel_counts[self._last_layer_idx]
        channel_frac = self._last_channel_idx / max(layer_channels - 1, 1)
        clean_norm = self.clean_accuracy / 100.0
        delta_norm = max(min(self._last_delta / 100.0, 1.0), -1.0)

        return torch.tensor(
            [layer_frac, channel_frac, clean_norm, delta_norm],
            dtype=torch.float32
        )
