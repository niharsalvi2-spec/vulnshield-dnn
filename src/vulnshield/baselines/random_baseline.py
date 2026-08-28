"""Random Channel Selection Baseline.

Randomly selects (layer_name, channel_idx) pairs within budget.
Serves as the minimum expected performance lower bound for discovery.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vulnshield.fault_injection.fault_injector import FaultInjector
from vulnshield.training.evaluator import evaluate_model


def run_random_baseline(
    model: nn.Module,
    dataloader: DataLoader,
    clean_accuracy: float,
    budget: int = 50,
    seed: int = 42,
    device: Optional[torch.device] = None
) -> List[Dict]:
    """Discover vulnerable channels via uniform random channel sampling.

    Within the given budget, randomly selects (layer_name, channel_idx) pairs,
    injects the fault, records ΔA, and returns results sorted by ΔA descending.

    Args:
        model: Pre-trained neural network.
        dataloader: Evaluation DataLoader.
        clean_accuracy: Pre-computed baseline clean accuracy (%).
        budget: Total number of fault evaluations allowed.
        seed: Random seed for reproducibility.
        device: Compute device.

    Returns:
        List of dicts with keys: layer_name, channel_idx, delta_accuracy, fault_accuracy.
        Sorted by delta_accuracy descending.
    """
    rng = random.Random(seed)
    injector = FaultInjector(model)
    injectable = injector.list_injectable_layers()

    # Build flat list of all (layer, channel) pairs
    all_pairs: List[Tuple[str, int]] = [
        (layer_name, c)
        for layer_name, n_channels in injectable
        for c in range(n_channels)
    ]

    # Sample `budget` unique pairs without replacement (if possible)
    n_sample = min(budget, len(all_pairs))
    sampled = rng.sample(all_pairs, k=n_sample)

    results = []
    for layer_name, channel_idx in sampled:
        with injector.inject([(layer_name, channel_idx)]):
            res = evaluate_model(model, dataloader, device=device)
        delta = clean_accuracy - res.accuracy
        results.append({
            "layer_name": layer_name,
            "channel_idx": channel_idx,
            "fault_accuracy": res.accuracy,
            "delta_accuracy": delta
        })

    results.sort(key=lambda d: d["delta_accuracy"], reverse=True)
    return results
