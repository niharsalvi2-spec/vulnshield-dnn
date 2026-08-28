"""Data Integrity and Batch Validation Utilities."""

from __future__ import annotations

from typing import Sequence, Tuple
import torch

from vulnshield.core.exceptions import DatasetError


def validate_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 10,
    expected_channels: int = 3,
    expected_size: Tuple[int, int] = (32, 32)
) -> None:
    """Validate format, dimensions, values, and finite bounds of a data batch.

    Args:
        images: Batch tensor of input images (B, C, H, W).
        labels: Batch tensor of ground-truth class labels (B,).
        num_classes: Total number of valid classes (default 10 for CIFAR-10).
        expected_channels: Expected channel dimension (default 3).
        expected_size: Expected spatial resolution (H, W).

    Raises:
        DatasetError: If batch validation criteria fail.
    """
    if not isinstance(images, torch.Tensor):
        raise DatasetError(f"Expected images to be a torch.Tensor, got {type(images).__name__}")
    if not isinstance(labels, torch.Tensor):
        raise DatasetError(f"Expected labels to be a torch.Tensor, got {type(labels).__name__}")

    if images.ndim != 4:
        raise DatasetError(f"Expected 4D image batch (B, C, H, W), got shape {images.shape}")
    if labels.ndim != 1:
        raise DatasetError(f"Expected 1D labels batch (B,), got shape {labels.shape}")

    batch_size = images.shape[0]
    if labels.shape[0] != batch_size:
        raise DatasetError(f"Batch size mismatch: images has {batch_size}, labels has {labels.shape[0]}")

    if images.shape[1] != expected_channels:
        raise DatasetError(f"Channel dimension mismatch: expected {expected_channels}, got {images.shape[1]}")
    if images.shape[2:] != expected_size:
        raise DatasetError(f"Spatial dimension mismatch: expected {expected_size}, got {images.shape[2:]}")

    if not torch.isfinite(images).all():
        raise DatasetError("Image batch contains NaN or infinite values.")

    if (labels < 0).any() or (labels >= num_classes).any():
        raise DatasetError(f"Labels contain values outside valid range [0, {num_classes - 1}]")


def validate_splits(train_indices: Sequence[int], val_indices: Sequence[int], total_expected: int = 50000) -> None:
    """Validate that train and validation index sets are disjoint and complete.

    Args:
        train_indices: Sequence of training set indices.
        val_indices: Sequence of validation set indices.
        total_expected: Total expected number of samples before split.

    Raises:
        DatasetError: If indices overlap or total count does not match.
    """
    train_set = set(train_indices)
    val_set = set(val_indices)

    overlap = train_set.intersection(val_set)
    if len(overlap) > 0:
        raise DatasetError(f"Train and validation splits overlap by {len(overlap)} samples.")

    total_actual = len(train_set) + len(val_set)
    if total_actual != total_expected:
        raise DatasetError(f"Split count mismatch: expected {total_expected}, got {total_actual}")
