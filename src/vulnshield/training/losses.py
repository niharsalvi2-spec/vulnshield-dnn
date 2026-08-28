"""Training Objectives, Loss Functions, and Accuracy Metrics."""

from __future__ import annotations

from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(nn.Module):
    """Standard Cross-Entropy Loss with optional label smoothing."""

    def __init__(self, label_smoothing: float = 0.0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, targets)


def calculate_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculate Top-1 classification accuracy percentage (0.0 to 100.0).

    Args:
        logits: Model predictions tensor (B, num_classes).
        targets: Ground-truth class labels (B,).

    Returns:
        Top-1 accuracy as a percentage float.
    """
    with torch.no_grad():
        preds = logits.argmax(dim=1)
        correct = (preds == targets).sum().item()
        total = targets.size(0)
        return (correct / total) * 100.0 if total > 0 else 0.0


def calculate_topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> Tuple[float, ...]:
    """Calculate Top-K classification accuracy for specified K values."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)
        if batch_size == 0:
            return tuple(0.0 for _ in topk)

        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True).item()
            res.append((correct_k / batch_size) * 100.0)
        return tuple(res)
