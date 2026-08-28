"""Weight Drift Regularization to Preserve Clean Baseline Representations."""

from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn


class WeightDriftRegularizer:
    """Penalizes deviation of fine-tuned model weights from baseline weights.

    Loss penalty: lambda * 0.5 * sum_i ||theta_i - theta_clean_i||^2
    """

    def __init__(self, baseline_model: nn.Module, lambda_drift: float = 1e-4):
        self.lambda_drift = lambda_drift
        self.baseline_weights: Dict[str, torch.Tensor] = {
            name: param.detach().clone()
            for name, param in baseline_model.named_parameters()
            if param.requires_grad
        }

    def compute_penalty(self, current_model: nn.Module) -> torch.Tensor:
        """Compute the L2 drift penalty tensor."""
        if self.lambda_drift <= 0.0:
            return torch.tensor(0.0)

        penalty = torch.tensor(0.0, device=next(current_model.parameters()).device)
        for name, param in current_model.named_parameters():
            if name in self.baseline_weights:
                base_w = self.baseline_weights[name].to(param.device)
                penalty = penalty + torch.sum((param - base_w) ** 2)

        return self.lambda_drift * 0.5 * penalty
