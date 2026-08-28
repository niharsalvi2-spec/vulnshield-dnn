"""Activation Magnitude Baseline.

Ranks channels by mean absolute activation magnitude across a calibration
mini-batch. Channels with the highest mean activation are considered the
most structurally important — and thus most vulnerable when zeroed.

Mathematically for channel c in layer l:
    score(c) = E_{x ~ D} [ ||X[:, c, :, :]||_1 / (H * W) ]
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vulnshield.fault_injection.fault_injector import FaultInjector


def run_activation_baseline(
    model: nn.Module,
    dataloader: DataLoader,
    budget: int = 50,
    device: Optional[torch.device] = None
) -> List[Dict]:
    """Score and rank channels by mean activation magnitude using forward hooks.

    Args:
        model: Pre-trained neural network.
        dataloader: Calibration DataLoader (a small evaluation batch is sufficient).
        budget: Maximum number of top-ranked channels to return.
        device: Compute device.

    Returns:
        List of dicts (layer_name, channel_idx, activation_score) sorted
        by activation_score descending.  Length = min(budget, total_channels).
    """
    dev = device or torch.device("cpu")
    model.eval()
    model.to(dev)

    injector = FaultInjector(model)
    injectable = injector.list_injectable_layers()

    # Register forward hooks to accumulate per-channel activation sums
    activation_sums: Dict[str, torch.Tensor] = {}
    activation_counts: Dict[str, int] = {}
    handles = []

    for layer_name, n_channels in injectable:
        # Resolve the layer
        layer = model
        for part in layer_name.split("."):
            layer = getattr(layer, part)

        # Closure to capture layer_name
        def make_hook(lname: str):
            def hook(module, inp, out: torch.Tensor):
                # out: (B, C, H, W)
                # mean abs per channel across spatial dims and batch
                with torch.no_grad():
                    mag = out.abs().mean(dim=(0, 2, 3)).cpu()  # (C,)
                if lname not in activation_sums:
                    activation_sums[lname] = torch.zeros(mag.shape)
                    activation_counts[lname] = 0
                activation_sums[lname] += mag
                activation_counts[lname] += 1
            return hook

        h = layer.register_forward_hook(make_hook(layer_name))
        handles.append(h)

    # Run forward pass over calibration data
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(dev)
            _ = model(images)
            break   # One batch is sufficient for activation statistics

    # Remove all hooks
    for h in handles:
        h.remove()

    # Build ranked list
    results: List[Dict] = []
    for layer_name, n_channels in injectable:
        if layer_name not in activation_sums:
            continue
        sums = activation_sums[layer_name]
        count = max(activation_counts[layer_name], 1)
        mean_acts = (sums / count).tolist()

        for c, score in enumerate(mean_acts):
            results.append({
                "layer_name": layer_name,
                "channel_idx": c,
                "activation_score": score
            })

    results.sort(key=lambda d: d["activation_score"], reverse=True)
    return results[:budget]
