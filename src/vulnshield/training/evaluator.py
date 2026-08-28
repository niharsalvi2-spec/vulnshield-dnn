"""Clean Model Evaluation Engine."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vulnshield.training.losses import calculate_accuracy, calculate_topk_accuracy
from vulnshield.utils.device import get_device


@dataclass
class EvaluationResult:
    """Structured container for model evaluation metrics."""
    loss: float
    accuracy: float
    top5_accuracy: float
    num_samples: int
    duration_seconds: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "loss": round(self.loss, 4),
            "accuracy": round(self.accuracy, 2),
            "top5_accuracy": round(self.top5_accuracy, 2),
            "num_samples": self.num_samples,
            "duration_seconds": round(self.duration_seconds, 2)
        }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Optional[nn.Module] = None,
    device: Optional[torch.device] = None
) -> EvaluationResult:
    """Evaluate model classification performance across a DataLoader.

    Args:
        model: PyTorch model to evaluate.
        dataloader: Test or validation DataLoader.
        criterion: Optional loss function (defaults to CrossEntropyLoss).
        device: Target compute device.

    Returns:
        EvaluationResult instance with loss, Top-1 and Top-5 accuracies.
    """
    dev = device or get_device()
    crit = criterion or nn.CrossEntropyLoss()

    model.eval()
    model.to(dev)

    total_loss = 0.0
    total_correct_top1 = 0
    total_correct_top5 = 0
    total_samples = 0

    start_time = time.time()

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(dev, non_blocking=True)
            targets = targets.to(dev, non_blocking=True)

            logits = model(images)
            loss = crit(logits, targets)

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            top1, top5 = calculate_topk_accuracy(logits, targets, topk=(1, 5))
            total_correct_top1 += (top1 / 100.0) * batch_size
            total_correct_top5 += (top5 / 100.0) * batch_size

    duration = time.time() - start_time
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    top1_acc = (total_correct_top1 / total_samples) * 100.0 if total_samples > 0 else 0.0
    top5_acc = (total_correct_top5 / total_samples) * 100.0 if total_samples > 0 else 0.0

    return EvaluationResult(
        loss=avg_loss,
        accuracy=top1_acc,
        top5_accuracy=top5_acc,
        num_samples=total_samples,
        duration_seconds=duration
    )
