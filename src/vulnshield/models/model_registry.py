"""Model Registry for Architecture Lookup and Registration."""

from __future__ import annotations

from typing import Callable, Dict, List, Type
import torch.nn as nn

from vulnshield.core.exceptions import ModelNotFoundError
from vulnshield.models.resnet.cifar_resnet18 import CIFARResNet18, resnet18
from vulnshield.models.vgg.cifar_vgg16 import CIFARVGG16, vgg16


_MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "resnet18": resnet18,
    "cifar_resnet18": resnet18,
    "vgg16": vgg16,
    "cifar_vgg16": vgg16,
}


def register_model(name: str, constructor: Callable[..., nn.Module]) -> None:
    """Register a custom model architecture constructor.

    Args:
        name: Unique string name of the architecture.
        constructor: Factory callable returning an nn.Module.
    """
    _MODEL_REGISTRY[name.lower().strip()] = constructor


def get_model_constructor(name: str) -> Callable[..., nn.Module]:
    """Retrieve model constructor by architecture name.

    Args:
        name: Name of the architecture.

    Returns:
        Constructor callable.

    Raises:
        ModelNotFoundError: If architecture name is not registered.
    """
    key = name.lower().strip()
    if key not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        raise ModelNotFoundError(f"Model architecture '{name}' not found in registry. Available: {available}")
    return _MODEL_REGISTRY[key]


def list_available_models() -> List[str]:
    """List all registered model architecture names."""
    return sorted(list(_MODEL_REGISTRY.keys()))
