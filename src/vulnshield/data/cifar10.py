"""CIFAR-10 Dataset Downloader and Native Dataset Wrapper."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import torchvision.datasets as datasets

from vulnshield.core.exceptions import DatasetError


def get_cifar10_dataset(
    root: Union[str, Path],
    train: bool = True,
    transform: Optional[Callable] = None,
    download: bool = True
) -> datasets.CIFAR10:
    """Retrieve or download the standard CIFAR-10 dataset.

    Args:
        root: Directory to store or locate raw CIFAR-10 dataset files.
        train: If True, retrieves training split (50,000 images), otherwise test split (10,000 images).
        transform: Optional torchvision transform to apply to images.
        download: If True, downloads the dataset if not present.

    Returns:
        torchvision.datasets.CIFAR10 instance.

    Raises:
        DatasetError: If dataset loading or downloading fails.
    """
    root_path = Path(root).resolve()
    root_path.mkdir(parents=True, exist_ok=True)

    try:
        dataset = datasets.CIFAR10(
            root=str(root_path),
            train=train,
            transform=transform,
            download=download
        )
        return dataset
    except Exception as e:
        raise DatasetError(f"Failed to load CIFAR-10 dataset from {root_path}: {e}") from e
