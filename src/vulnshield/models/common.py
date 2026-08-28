"""Common Neural Network Modules and Architecture Utilities."""

from __future__ import annotations

from typing import List, Tuple
import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    """Standard Convolution + Batch Normalization + ReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Calculate the total and trainable parameter counts for a PyTorch model.

    Args:
        model: PyTorch model instance.

    Returns:
        Tuple of (total_params, trainable_params).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_named_conv_layers(model: nn.Module) -> List[Tuple[str, nn.Conv2d]]:
    """Enumerate all 2D convolutional layers in execution order.

    Args:
        model: Target PyTorch neural network.

    Returns:
        List of (layer_name, nn.Conv2d module) pairs.
    """
    conv_layers: List[Tuple[str, nn.Conv2d]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))
    return conv_layers
