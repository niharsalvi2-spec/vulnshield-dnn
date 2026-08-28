"""Multi-Worker DataLoader Factory for CIFAR-10."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
import torch
from torch.utils.data import DataLoader, Subset

from vulnshield.data.cifar10 import get_cifar10_dataset
from vulnshield.data.transforms import get_train_transforms, get_test_transforms
from vulnshield.data.splits import (
    TransformedSubset,
    create_stratified_train_val_split,
    create_fixed_eval_indices
)
from vulnshield.utils.config import ConfigDict
from vulnshield.utils.reproducibility import get_generator, seed_worker


@dataclass
class DataLoadersContainer:
    """Container holding all standard DataLoaders for an experiment run."""
    train: DataLoader
    val: DataLoader
    test: DataLoader
    eval_fault: DataLoader


def build_cifar10_dataloaders(
    data_dir: Union[str, Path],
    config: Optional[ConfigDict] = None,
    seed: int = 42
) -> DataLoadersContainer:
    """Construct deterministic training, validation, test, and fault-eval DataLoaders.

    Args:
        data_dir: Directory containing or targeted for CIFAR-10 raw dataset.
        config: Optional configuration dictionary.
        seed: Random seed for deterministic data loading.

    Returns:
        DataLoadersContainer instance with train, val, test, and eval_fault loaders.
    """
    # Extract config parameters with robust defaults
    train_bs = 128
    val_bs = 128
    test_bs = 128
    eval_bs = 128
    num_workers = 4
    pin_memory = True
    val_ratio = 0.1
    eval_samples = 1000

    if config is not None:
        dl_cfg = config.get("dataloader", {})
        train_bs = dl_cfg.get("train_batch_size", train_bs)
        val_bs = dl_cfg.get("val_batch_size", val_bs)
        test_bs = dl_cfg.get("test_batch_size", test_bs)
        eval_bs = dl_cfg.get("eval_fault_batch_size", eval_bs)
        num_workers = dl_cfg.get("num_workers", num_workers)
        pin_memory = dl_cfg.get("pin_memory", pin_memory)
        
        split_cfg = config.get("splits", {})
        eval_samples = split_cfg.get("eval_fault_batch_samples", eval_samples)

    # Disable pin_memory if CUDA is unavailable
    if not torch.cuda.is_available():
        pin_memory = False

    # 1. Load Raw Datasets (without transforms so we can apply specific split transforms)
    train_val_raw = get_cifar10_dataset(root=data_dir, train=True, transform=None, download=True)
    test_raw = get_cifar10_dataset(root=data_dir, train=False, transform=None, download=True)

    # 2. Build Transforms
    train_transform = get_train_transforms(augment=True)
    test_transform = get_test_transforms()

    # 3. Create Stratified Train / Val Split
    train_idx, val_idx = create_stratified_train_val_split(
        targets=train_val_raw.targets,
        val_ratio=val_ratio,
        seed=seed
    )

    train_subset = TransformedSubset(train_val_raw, train_idx, transform=train_transform)
    val_subset = TransformedSubset(train_val_raw, val_idx, transform=test_transform)

    # 4. Create Fixed Fault-Evaluation Subset (from test set)
    eval_idx = create_fixed_eval_indices(
        targets=test_raw.targets,
        num_samples=eval_samples,
        seed=seed
    )
    test_subset = TransformedSubset(test_raw, range(len(test_raw)), transform=test_transform)
    eval_fault_subset = TransformedSubset(test_raw, eval_idx, transform=test_transform)

    # 5. Build Seeded Generators
    train_gen = get_generator(seed)

    # 6. Instantiate DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=train_bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=train_gen,
        drop_last=False
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=test_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    eval_fault_loader = DataLoader(
        eval_fault_subset,
        batch_size=eval_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return DataLoadersContainer(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        eval_fault=eval_fault_loader
    )
