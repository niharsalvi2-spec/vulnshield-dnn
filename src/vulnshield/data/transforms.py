"""CIFAR-10 Data Transformations and Preprocessing Pipelines."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import torch
import torchvision.transforms as T


CIFAR10_MEAN: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616)


def get_train_transforms(
    mean: Sequence[float] = CIFAR10_MEAN,
    std: Sequence[float] = CIFAR10_STD,
    augment: bool = True
) -> T.Compose:
    """Build preprocessing and augmentation pipeline for training data.

    Args:
        mean: Sequence of per-channel normalization means.
        std: Sequence of per-channel normalization standard deviations.
        augment: If True, applies random cropping and horizontal flipping.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    transform_list = []
    if augment:
        transform_list.extend([
            T.RandomCrop(32, padding=4, padding_mode="reflect"),
            T.RandomHorizontalFlip(p=0.5)
        ])
    transform_list.extend([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
    return T.Compose(transform_list)


def get_test_transforms(
    mean: Sequence[float] = CIFAR10_MEAN,
    std: Sequence[float] = CIFAR10_STD
) -> T.Compose:
    """Build deterministic preprocessing pipeline for validation and evaluation data.

    Args:
        mean: Sequence of per-channel normalization means.
        std: Sequence of per-channel normalization standard deviations.

    Returns:
        torchvision.transforms.Compose pipeline.
    """
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])


def get_inverse_transforms(
    mean: Sequence[float] = CIFAR10_MEAN,
    std: Sequence[float] = CIFAR10_STD
) -> T.Compose:
    """Build inverse normalization transform to convert normalized tensors back to RGB image range [0, 1]."""
    inv_mean = [-m / s for m, s in zip(mean, std)]
    inv_std = [1.0 / s for s in std]
    return T.Compose([
        T.Normalize(mean=inv_mean, std=inv_std)
    ])
