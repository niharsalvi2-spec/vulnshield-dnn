"""Physical Bit-Flip Hardware Fault Simulation (IEEE 754 Float32).

Simulates radiation-induced Single Event Upsets (SEUs) or memory bit-flips
in 32-bit floating-point weights and activation registers:
  - Sign bit: bit 31 (inverts sign)
  - Exponent bits: bits 23-30 (causes catastrophic order-of-magnitude surges/underflows)
  - Mantissa bits: bits 0-22 (minor numerical perturbations)
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from vulnshield.training.evaluator import evaluate_model
from vulnshield.models.common import get_named_conv_layers


BIT_POSITIONS = {
    "sign": 31,
    "exponent": 27,   # Middle of the 8-bit exponent field [23..30]
    "mantissa": 10    # Middle of the 23-bit mantissa field [0..22]
}


def flip_float32_bit(tensor: torch.Tensor, bit_position: int) -> torch.Tensor:
    """Flip a specific bit in an IEEE 754 32-bit floating-point tensor via view casting.

    Args:
        tensor: Float32 tensor.
        bit_position: Bit index in [0, 31].

    Returns:
        New Float32 tensor with the designated bit inverted.
    """
    # Cast float32 view to int32 without reinterpreting values
    int_view = tensor.view(torch.int32).clone()
    mask = 1 << bit_position
    int_view = int_view ^ mask
    return int_view.view(torch.float32)


def evaluate_bit_flip_robustness(
    model: nn.Module,
    dataloader: DataLoader,
    target_bits: Sequence[str] = ("sign", "exponent", "mantissa"),
    flips_per_layer: int = 10,
    seed: int = 42,
    device: Optional[torch.device] = None
) -> Dict[str, float]:
    """Evaluate classification accuracy under single bit-flips across Conv2d weights.

    Args:
        model: Evaluated neural network.
        dataloader: Test or evaluation DataLoader.
        target_bits: Bit types to evaluate ('sign', 'exponent', 'mantissa').
        flips_per_layer: Number of random weights perturbed per layer.
        seed: Random seed.
        device: Compute device.

    Returns:
        Dict mapping bit_type to average accuracy under single bit-flip.
    """
    dev = device or torch.device("cpu")
    model.eval()
    model.to(dev)

    conv_layers = get_named_conv_layers(model)
    results_by_bit: Dict[str, float] = {}

    rng = random.Random(seed)

    for bit_type in target_bits:
        bit_idx = BIT_POSITIONS.get(bit_type, 31)
        trial_accuracies = []

        for name, layer in conv_layers:
            w_shape = layer.weight.shape
            total_elements = layer.weight.numel()

            # Sample random weight indices
            sample_indices = rng.sample(range(total_elements), k=min(flips_per_layer, total_elements))

            for flat_idx in sample_indices:
                orig_flat = layer.weight.data.view(-1).clone()

                # Flip single bit
                flipped_val = flip_float32_bit(orig_flat[flat_idx], bit_idx)
                orig_flat[flat_idx] = flipped_val

                # Replace layer weights temporarily
                layer.weight.data = orig_flat.view(w_shape)

                # Evaluate under bit-flip
                with torch.no_grad():
                    res = evaluate_model(model, dataloader, device=dev)
                    trial_accuracies.append(res.accuracy)

                # Restore original weight
                orig_flat[flat_idx] = flip_float32_bit(flipped_val, bit_idx)
                layer.weight.data = orig_flat.view(w_shape)

        avg_acc = sum(trial_accuracies) / max(len(trial_accuracies), 1)
        results_by_bit[bit_type] = avg_acc

    return results_by_bit
