"""Clean Accuracy Evaluation and Baseline Retention Check."""

from __future__ import annotations

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vulnshield.training.evaluator import evaluate_model, EvaluationResult


def evaluate_clean_preservation(
    model: nn.Module,
    dataloader: DataLoader,
    baseline_clean_accuracy: float,
    device: Optional[torch.device] = None,
    max_tolerable_drop: float = 1.0
) -> Tuple[EvaluationResult, float, bool]:
    """Verify that model clean accuracy has not degraded by more than max_tolerable_drop.

    Args:
        model: Evaluated neural network.
        dataloader: Test or evaluation DataLoader.
        baseline_clean_accuracy: Original clean unhardened model accuracy (%).
        device: Compute device.
        max_tolerable_drop: Maximum acceptable drop in percentage points (default 1.0%).

    Returns:
        Tuple of (EvaluationResult, accuracy_drop, is_within_tolerance).
    """
    res = evaluate_model(model, dataloader, device=device)
    drop = baseline_clean_accuracy - res.accuracy
    passed = drop <= max_tolerable_drop
    return res, drop, passed
