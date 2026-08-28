"""Optimizer Factory and Builder for Model Training."""

from __future__ import annotations

from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import torch.optim as optim

from vulnshield.core.exceptions import ConfigurationError


def build_optimizer(
    model: nn.Module,
    name: str = "sgd",
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    nesterov: bool = True,
    **kwargs
) -> optim.Optimizer:
    """Instantiate a configured PyTorch optimizer for model parameters.

    Args:
        model: Target neural network model.
        name: Name of optimizer ('sgd', 'adam', 'adamw').
        lr: Base learning rate.
        momentum: Momentum factor for SGD.
        weight_decay: L2 regularization penalty factor.
        nesterov: Enable Nesterov momentum for SGD.

    Returns:
        torch.optim.Optimizer instance.

    Raises:
        ConfigurationError: If optimizer name is unsupported.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    opt_name = name.lower().strip()

    if opt_name == "sgd":
        return optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov
        )
    elif opt_name == "adam":
        return optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    elif opt_name == "adamw":
        return optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
    else:
        raise ConfigurationError(f"Unsupported optimizer: '{name}'. Supported: ['sgd', 'adam', 'adamw']")
