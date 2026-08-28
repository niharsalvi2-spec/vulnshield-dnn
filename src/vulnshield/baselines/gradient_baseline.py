"""Gradient / Taylor First-Order Sensitivity Baseline.

Approximates channel importance using first-order Taylor expansion:

    score(c) = |∇_{X_c} L · X_c|
             = E_{x ~ D} [ ||grad_c * act_c||_1 / (H * W) ]

where:
    - act_c  = output activations of channel c  (forward pass)
    - grad_c = gradient of loss w.r.t. act_c    (backward pass)

Channels with high |grad × act| have a large first-order effect on the loss
when zeroed — they are the most sensitive targets.

Reference:
    Molchanov et al. (2017) "Pruning Convolutional Neural Networks for
    Resource Efficient Inference" — arXiv:1611.06440
"""

from __future__ import annotations

from typing import Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vulnshield.fault_injection.fault_injector import FaultInjector


def run_gradient_baseline(
    model: nn.Module,
    dataloader: DataLoader,
    budget: int = 50,
    device: Optional[torch.device] = None
) -> List[Dict]:
    """Score channels by Taylor first-order sensitivity.

    Performs ONE forward+backward pass over a calibration mini-batch,
    captures (activation, gradient) pairs per channel, and computes
    the Taylor importance score.

    Args:
        model: Pre-trained neural network (weights unchanged after this call).
        dataloader: Calibration DataLoader.
        budget: Maximum number of top-ranked channels to return.
        device: Compute device.

    Returns:
        List of dicts (layer_name, channel_idx, gradient_score) sorted
        by gradient_score descending.
    """
    dev = device or torch.device("cpu")
    model.eval()
    model.to(dev)

    injector = FaultInjector(model)
    injectable = injector.list_injectable_layers()

    activations: Dict[str, torch.Tensor] = {}
    gradients: Dict[str, torch.Tensor] = {}
    handles = []

    for layer_name, _ in injectable:
        layer = model
        for part in layer_name.split("."):
            layer = getattr(layer, part)

        def make_fwd_hook(lname: str):
            def hook(module, inp, out: torch.Tensor):
                activations[lname] = out  # keep gradient-connected tensor
            return hook

        def make_bwd_hook(lname: str):
            def hook(module, grad_in, grad_out):
                # grad_out[0]: gradient of loss w.r.t. layer output (B, C, H, W)
                if grad_out[0] is not None:
                    gradients[lname] = grad_out[0].detach()
            return hook

        h1 = layer.register_forward_hook(make_fwd_hook(layer_name))
        h2 = layer.register_full_backward_hook(make_bwd_hook(layer_name))
        handles.extend([h1, h2])

    criterion = nn.CrossEntropyLoss()

    # One forward + backward pass
    images, labels = next(iter(dataloader))
    images = images.to(dev)
    labels = labels.to(dev)

    model.zero_grad()
    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()

    # Remove hooks
    for h in handles:
        h.remove()

    # Compute Taylor score: mean |grad * act| per channel
    results: List[Dict] = []
    for layer_name, n_channels in injectable:
        if layer_name not in activations or layer_name not in gradients:
            continue
        act = activations[layer_name].detach()   # (B, C, H, W)
        grad = gradients[layer_name]             # (B, C, H, W)

        # Taylor score per channel: mean over batch and spatial dims
        score_per_channel = (grad * act).abs().mean(dim=(0, 2, 3))   # (C,)

        for c in range(n_channels):
            results.append({
                "layer_name": layer_name,
                "channel_idx": c,
                "gradient_score": score_per_channel[c].item()
            })

    # Clean up gradients to restore model to eval state
    model.zero_grad()

    results.sort(key=lambda d: d["gradient_score"], reverse=True)
    return results[:budget]
