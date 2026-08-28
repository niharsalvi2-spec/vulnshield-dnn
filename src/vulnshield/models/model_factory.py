"""Model Factory for Model Creation, Checkpointing, and State Loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn

from vulnshield.core.exceptions import ModelNotFoundError
from vulnshield.models.model_registry import get_model_constructor
from vulnshield.utils.device import get_device, to_device


def create_model(
    name: str,
    num_classes: int = 10,
    checkpoint_path: Optional[Union[str, Path]] = None,
    device: Optional[Union[str, torch.device]] = None,
    **kwargs
) -> nn.Module:
    """Instantiate and optionally load weights for a target model architecture.

    Args:
        name: Name of the architecture (e.g. 'resnet18', 'vgg16').
        num_classes: Number of output classification classes.
        checkpoint_path: Optional path to a trained checkpoint (.pt/.pth file).
        device: Target compute device.

    Returns:
        Configured nn.Module instance.
    """
    constructor = get_model_constructor(name)
    model = constructor(num_classes=num_classes, **kwargs)

    dev = get_device(str(device) if device is not None else None)
    model = model.to(dev)

    if checkpoint_path is not None:
        load_model_weights(model, checkpoint_path, device=dev)

    return model


def save_checkpoint(
    model: nn.Module,
    path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """Save model state dictionary, optimizer state, and metadata to a checkpoint file.

    Args:
        model: PyTorch model instance.
        path: Target file path (.pt).
        optimizer: Optional optimizer to save state for.
        epoch: Optional epoch number.
        metrics: Optional dictionary of evaluation metrics.
        extra: Additional metadata.
    """
    save_path = Path(path).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = model.state_dict()
    payload: Dict[str, Any] = {
        "model_state_dict": state_dict,
        "epoch": epoch,
        "metrics": metrics or {},
        "extra": extra or {}
    }

    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(payload, str(save_path))


def load_model_weights(
    model: nn.Module,
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """Load weights from a checkpoint file into a model instance.

    Args:
        model: PyTorch model instance to load weights into.
        checkpoint_path: Path to checkpoint file.
        device: Target device for loaded tensors.
        strict: If True, enforces exact match of state dict keys.

    Returns:
        Checkpoint metadata dictionary.

    Raises:
        ModelNotFoundError: If checkpoint file does not exist or fails to load.
    """
    ckpt_path = Path(checkpoint_path).resolve()
    if not ckpt_path.exists():
        raise ModelNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    dev = device or get_device()
    try:
        payload = torch.load(str(ckpt_path), map_location=dev, weights_only=False)
    except Exception as e:
        raise ModelNotFoundError(f"Failed to load checkpoint file {ckpt_path}: {e}") from e

    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        metadata = {k: v for k, v in payload.items() if k != "model_state_dict"}
    else:
        # Direct state dict
        state_dict = payload
        metadata = {}

    model.load_state_dict(state_dict, strict=strict)
    return metadata


# Alias for backward compatibility
build_model = create_model

