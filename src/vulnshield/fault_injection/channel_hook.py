"""Low-Level Stuck-at-Zero Channel Hook for Forward Pass Fault Injection.

A channel hook zeros ALL activations in a single output channel of a Conv2d
layer during the forward pass, without modifying any model weights.

Mathematically:
    For fault on layer l, channel c:
        X[:, c, :, :] = 0   (applied to the Conv2d output tensor)

This is applied via PyTorch's register_forward_hook mechanism so:
  - Model weights are NEVER modified
  - The hook is fully removable after evaluation
  - Multiple hooks can be registered simultaneously for interaction studies
"""

from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn


class StuckAtZeroHook:
    """Registers a stuck-at-zero fault on a single output channel of a Conv2d layer.

    Attributes:
        layer: The target nn.Conv2d module.
        channel_idx: Output channel index to zero out.
        layer_name: Human-readable identifier for the layer.
        _handle: Internal PyTorch hook handle for removal.
    """

    def __init__(
        self,
        layer: nn.Conv2d,
        channel_idx: int,
        layer_name: str = ""
    ):
        if not isinstance(layer, nn.Conv2d):
            raise TypeError(f"StuckAtZeroHook requires nn.Conv2d, got {type(layer).__name__}")
        if channel_idx < 0 or channel_idx >= layer.out_channels:
            raise IndexError(
                f"channel_idx={channel_idx} is out of range for layer with "
                f"out_channels={layer.out_channels} (name='{layer_name}')"
            )

        self.layer = layer
        self.channel_idx = channel_idx
        self.layer_name = layer_name
        self._handle: Optional[torch.utils.hooks.RemovableHook] = None
        self._active = False

    def _hook_fn(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor, ...],
        output: torch.Tensor
    ) -> torch.Tensor:
        """PyTorch forward hook: zeros the target output channel in-place.

        The hook zeroes activations without creating a new tensor, ensuring
        minimal memory overhead and correct gradient flow blocking for
        subsequent layers that would receive zero input from this channel.
        """
        output[:, self.channel_idx, :, :] = 0.0
        return output

    def register(self) -> "StuckAtZeroHook":
        """Register the forward hook on the target layer.

        Returns:
            Self for fluent chaining.

        Raises:
            RuntimeError: If the hook has already been registered.
        """
        if self._active:
            raise RuntimeError(
                f"Hook already registered on layer '{self.layer_name}', "
                f"channel {self.channel_idx}. Call remove() first."
            )
        self._handle = self.layer.register_forward_hook(self._hook_fn)
        self._active = True
        return self

    def remove(self) -> None:
        """Remove the forward hook, restoring normal layer behaviour."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def __repr__(self) -> str:
        return (
            f"StuckAtZeroHook("
            f"layer='{self.layer_name}', "
            f"channel={self.channel_idx}/{self.layer.out_channels}, "
            f"active={self._active})"
        )

    def __enter__(self) -> "StuckAtZeroHook":
        self.register()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.remove()
