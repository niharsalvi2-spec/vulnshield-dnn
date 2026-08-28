"""Evaluation across Known Discovered, Unseen Random, and Multi-Fault Sets."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from vulnshield.fault_injection.fault_injector import FaultInjector, FaultSpec
from vulnshield.training.evaluator import evaluate_model


def evaluate_channel_fault_set(
    model: nn.Module,
    channels: Sequence[FaultSpec],
    dataloader: DataLoader,
    clean_accuracy: float,
    device: Optional[torch.device] = None,
    desc: str = "Evaluating Fault Set"
) -> Tuple[float, float, List[Dict]]:
    """Evaluate mean accuracy and per-channel drops across a set of target channels.

    Returns:
        Tuple of (mean_fault_accuracy, mean_accuracy_drop, detailed_results_list).
    """
    dev = device or torch.device("cpu")
    injector = FaultInjector(model)
    detailed = []
    accuracies = []

    for ch in tqdm(channels, desc=desc, leave=False):
        with injector.inject([ch]):
            res = evaluate_model(model, dataloader, device=dev)
        drop = clean_accuracy - res.accuracy
        accuracies.append(res.accuracy)
        detailed.append({
            "layer_name": ch[0],
            "channel_idx": ch[1],
            "fault_accuracy": res.accuracy,
            "accuracy_drop": drop
        })

    mean_acc = sum(accuracies) / max(len(accuracies), 1)
    mean_drop = clean_accuracy - mean_acc
    return mean_acc, mean_drop, detailed


def evaluate_unseen_channel_generalization(
    model: nn.Module,
    protected_channels: Sequence[FaultSpec],
    all_injectable_layers: Sequence[Tuple[str, int]],
    dataloader: DataLoader,
    clean_accuracy: float,
    num_unseen_samples: int = 50,
    seed: int = 42,
    device: Optional[torch.device] = None
) -> Tuple[float, float, List[Dict]]:
    """Evaluate generalization robustness on channels that were NEVER used during training."""
    protected_set = set(protected_channels)
    all_unseen: List[FaultSpec] = [
        (layer_name, c)
        for layer_name, n_ch in all_injectable_layers
        for c in range(n_ch)
        if (layer_name, c) not in protected_set
    ]

    rng = random.Random(seed)
    sampled_unseen = rng.sample(all_unseen, k=min(num_unseen_samples, len(all_unseen)))

    return evaluate_channel_fault_set(
        model=model,
        channels=sampled_unseen,
        dataloader=dataloader,
        clean_accuracy=clean_accuracy,
        device=device,
        desc="Unseen Fault Generalization"
    )


def evaluate_simultaneous_multi_faults(
    model: nn.Module,
    all_injectable_layers: Sequence[Tuple[str, int]],
    dataloader: DataLoader,
    fault_counts: Sequence[int] = (2, 3, 5),
    trials_per_count: int = 20,
    seed: int = 42,
    device: Optional[torch.device] = None
) -> Dict[int, float]:
    """Stress-test model under increasing counts of simultaneous random channel faults."""
    dev = device or torch.device("cpu")
    injector = FaultInjector(model)
    rng = random.Random(seed)

    all_channels: List[FaultSpec] = [
        (layer_name, c) for layer_name, n_ch in all_injectable_layers for c in range(n_ch)
    ]

    results_by_count: Dict[int, float] = {}

    for k in fault_counts:
        trial_accs = []
        for _ in range(trials_per_count):
            sampled = rng.sample(all_channels, k=min(k, len(all_channels)))
            with injector.inject(sampled):
                res = evaluate_model(model, dataloader, device=dev)
            trial_accs.append(res.accuracy)
        results_by_count[k] = sum(trial_accs) / max(len(trial_accs), 1)

    return results_by_count
