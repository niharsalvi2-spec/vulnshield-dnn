"""General Dataset Interface and Abstraction for VulnShield-DNN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class DatasetMetadata:
    """Immutable metadata descriptor for an evaluation dataset."""
    name: str
    num_classes: int
    image_shape: Tuple[int, int, int]
    class_names: List[str]
    train_count: int
    val_count: int
    test_count: int


CIFAR10_METADATA = DatasetMetadata(
    name="cifar10",
    num_classes=10,
    image_shape=(3, 32, 32),
    class_names=[
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ],
    train_count=45000,
    val_count=5000,
    test_count=10000
)
