"""Dataset Splitting and Subset Generation for VulnShield-DNN.

Provides deterministic, stratified train/validation partitioning and fixed evaluation subsets.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from vulnshield.core.exceptions import DatasetError


class TransformedSubset(Dataset):
    """A PyTorch Subset that applies a specific transform override to underlying samples."""

    def __init__(self, dataset: Dataset, indices: Sequence[int], transform: Optional[Callable] = None):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, target = self.dataset[self.indices[idx]]
        if self.transform is not None:
            # If the underlying dataset already transformed image into a PIL or tensor,
            # we apply our custom transform
            if not isinstance(image, torch.Tensor):
                image = self.transform(image)
        return image, target

    def __len__(self) -> int:
        return len(self.indices)


def create_stratified_train_val_split(
    targets: Sequence[int],
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """Generate stratified indices for training and validation sets.

    Args:
        targets: Sequence of integer class labels for all training samples.
        val_ratio: Proportion of samples allocated to validation (e.g. 0.1 for 5,000 / 50,000).
        seed: Random seed for deterministic index shuffling.

    Returns:
        Tuple of (train_indices, val_indices).
    """
    targets_arr = np.array(targets)
    unique_classes = np.unique(targets_arr)
    rng = np.random.RandomState(seed)

    train_indices: List[int] = []
    val_indices: List[int] = []

    for cls in unique_classes:
        cls_indices = np.where(targets_arr == cls)[0]
        rng.shuffle(cls_indices)
        val_count = int(np.round(len(cls_indices) * val_ratio))
        val_indices.extend(cls_indices[:val_count].tolist())
        train_indices.extend(cls_indices[val_count:].tolist())

    # Final shuffle to avoid class clustering
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices


def create_fixed_eval_indices(
    targets: Sequence[int],
    num_samples: int = 1000,
    seed: int = 42
) -> List[int]:
    """Generate balanced, fixed subset indices for fast fault-injection evaluation batches.

    Args:
        targets: Sequence of integer class labels (e.g. test set targets).
        num_samples: Total number of samples in the fixed evaluation subset.
        seed: Random seed for determinism.

    Returns:
        List of selected sample indices.
    """
    targets_arr = np.array(targets)
    unique_classes = np.unique(targets_arr)
    num_classes = len(unique_classes)
    samples_per_class = num_samples // num_classes
    rng = np.random.RandomState(seed)

    eval_indices: List[int] = []
    for cls in unique_classes:
        cls_indices = np.where(targets_arr == cls)[0]
        rng.shuffle(cls_indices)
        eval_indices.extend(cls_indices[:samples_per_class].tolist())

    rng.shuffle(eval_indices)
    return eval_indices
