"""Fault-Aware Composite Loss Function."""

from __future__ import annotations

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class FaultAwareLoss(nn.Module):
    """Composite training loss: L_total = alpha * L_clean + beta * L_fault.

    Args:
        alpha: Weight for clean classification loss.
        beta: Weight for fault-injected classification loss.
        label_smoothing: Optional cross-entropy label smoothing.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(
        self,
        clean_logits: torch.Tensor,
        fault_logits: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute composite fault-aware loss.

        Args:
            clean_logits: Predictions from normal un-hooked forward pass.
            fault_logits: Predictions from forward pass with faulted channel(s).
            targets: Ground-truth class labels.

        Returns:
            Tuple of (total_loss, clean_loss, fault_loss).
        """
        l_clean = self.criterion(clean_logits, targets)
        l_fault = self.criterion(fault_logits, targets)
        total_loss = self.alpha * l_clean + self.beta * l_fault
        return total_loss, l_clean, l_fault
