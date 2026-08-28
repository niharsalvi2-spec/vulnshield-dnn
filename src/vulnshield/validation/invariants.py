"""Scientific Invariant Verification Suite for Vulnerability & Fault Modeling."""

from __future__ import annotations

from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vulnshield.fault_injection.fault_injector import FaultInjector
from vulnshield.fault_injection.channel_hook import StuckAtZeroHook
from vulnshield.evaluation.bit_flip import flip_float32_bit
from vulnshield.discovery.action_mapper import ActionMapper


def verify_weight_immutability(
    model: nn.Module,
    injector: FaultInjector,
    dummy_input: torch.Tensor
) -> bool:
    """Invariant 1: Fault injection via forward hooks must NEVER modify underlying layer weights."""
    orig_params = [p.clone().detach() for p in model.parameters()]

    injectable = injector.list_injectable_layers()
    if not injectable:
        return True

    layer_name, n_ch = injectable[0]
    # Inject fault and run forward pass
    with injector.inject([(layer_name, 0)]):
        _ = model(dummy_input)

    # Post-injection parameter comparison
    post_params = list(model.parameters())
    for p_orig, p_post in zip(orig_params, post_params):
        if not torch.equal(p_orig, p_post.data):
            return False
    return True


def verify_bit_flip_reversibility(
    test_tensor: torch.Tensor,
    bit_position: int
) -> bool:
    """Invariant 2: Inverting bit_position twice must perfectly restore original float32 tensor."""
    flipped_once = flip_float32_bit(test_tensor, bit_position)
    flipped_twice = flip_float32_bit(flipped_once, bit_position)
    # Check bitwise representation equality
    return bool(torch.equal(test_tensor.view(torch.int32), flipped_twice.view(torch.int32)))


def verify_action_mapper_soundness(
    mapper: ActionMapper,
    grid_resolution: Optional[int] = None
) -> Dict[str, bool]:
    """Invariant 3: Continuous 2D action grid [-1, 1]^2 must always produce valid, reachable channel indices."""
    res = grid_resolution or max(len(mapper.layer_channel_counts) * 3, 60)
    vals = torch.linspace(-1.0, 1.0, res)
    all_in_bounds = True
    reachable_layers = set()

    for a1 in vals:
        for a2 in vals:
            action = torch.tensor([a1.item(), a2.item()])
            layer_name, ch_idx = mapper.decode(action)
            reachable_layers.add(layer_name)

            # Check layer existence and bounds
            match = next((c for n, c in mapper.layer_channel_counts if n == layer_name), None)
            if match is None or ch_idx < 0 or ch_idx >= match:
                all_in_bounds = False

    return {
        "all_in_bounds": all_in_bounds,
        "all_layers_reachable": len(reachable_layers) == len(mapper.layer_channel_counts)
    }
