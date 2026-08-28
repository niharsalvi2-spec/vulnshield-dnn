"""Fault Injector: Registration Manager for Single and Multi-Channel Faults.

Provides FaultInjector, a high-level context manager that:
  1. Resolves the target Conv2d layer by its full qualified name
     (e.g. "layer2.0.conv1" for ResNet-18)
  2. Registers one StuckAtZeroHook per (layer_name, channel_idx) pair
  3. Cleans up ALL hooks deterministically on exit — even if an exception
     is raised mid-evaluation
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Generator, List, Optional, Sequence, Tuple
import torch.nn as nn

from vulnshield.core.exceptions import FaultInjectionError
from vulnshield.fault_injection.channel_hook import StuckAtZeroHook


FaultSpec = Tuple[str, int]   # (layer_name, channel_idx)


def _resolve_layer(model: nn.Module, layer_name: str) -> nn.Conv2d:
    """Walk the model module tree to retrieve a Conv2d layer by dotted name.

    Args:
        model: Root nn.Module.
        layer_name: Dotted qualified name, e.g. ``"layer2.0.conv1"``.

    Returns:
        The matching nn.Conv2d layer.

    Raises:
        FaultInjectionError: If the name is not found or is not a Conv2d.
    """
    module: nn.Module = model
    for part in layer_name.split("."):
        try:
            module = getattr(module, part)
        except AttributeError:
            raise FaultInjectionError(
                f"Layer name segment '{part}' not found in model while "
                f"resolving '{layer_name}'. Check layer names with "
                f"get_named_conv_layers()."
            )
    if not isinstance(module, nn.Conv2d):
        raise FaultInjectionError(
            f"Layer '{layer_name}' is {type(module).__name__}, not nn.Conv2d. "
            f"Only Conv2d layers are supported for fault injection."
        )
    return module


class FaultInjector:
    """Manages registration and removal of stuck-at-zero faults on a model.

    Example — single fault:
        injector = FaultInjector(model)
        with injector.inject([("layer2.0.conv1", 42)]):
            out = model(x)      # channel 42 of layer2.0.conv1 is zeroed

    Example — multi-channel simultaneous fault:
        with injector.inject([("layer2.0.conv1", 10), ("layer3.1.conv2", 5)]):
            out = model(x)
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._active_hooks: List[StuckAtZeroHook] = []

    @contextmanager
    def inject(self, fault_specs: Sequence[FaultSpec]) -> Generator[None, None, None]:
        """Context manager that applies the given fault specifications during the block.

        Args:
            fault_specs: List of (layer_name, channel_idx) pairs defining faults.

        Yields:
            None — the block executes with all hooks registered.

        Raises:
            FaultInjectionError: On layer resolution or hook registration failure.
        """
        hooks: List[StuckAtZeroHook] = []
        try:
            for layer_name, channel_idx in fault_specs:
                layer = _resolve_layer(self.model, layer_name)
                hook = StuckAtZeroHook(layer, channel_idx, layer_name)
                hook.register()
                hooks.append(hook)
            yield
        finally:
            for hook in hooks:
                hook.remove()

    def get_layer_channel_count(self, layer_name: str) -> int:
        """Return the number of output channels for a named Conv2d layer.

        Args:
            layer_name: Dotted qualified layer name.

        Returns:
            Number of output channels (int).
        """
        layer = _resolve_layer(self.model, layer_name)
        return layer.out_channels

    def list_injectable_layers(self) -> List[Tuple[str, int]]:
        """Enumerate all Conv2d layers available for fault injection.

        Returns:
            List of (layer_name, out_channels) tuples in model execution order.
        """
        result = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                result.append((name, module.out_channels))
        return result
