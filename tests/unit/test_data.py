"""Unit and Integration Tests for VulnShield-DNN Data Layer."""

from pathlib import Path
import pytest
import numpy as np
import torch
from PIL import Image

from vulnshield.core.exceptions import DatasetError
from vulnshield.data.transforms import (
    get_train_transforms,
    get_test_transforms,
    get_inverse_transforms,
    CIFAR10_MEAN,
    CIFAR10_STD
)
from vulnshield.data.splits import (
    create_stratified_train_val_split,
    create_fixed_eval_indices,
    TransformedSubset
)
from vulnshield.data.validation import validate_batch, validate_splits
from vulnshield.data.datasets import CIFAR10_METADATA


@pytest.mark.unit
class TestTransforms:
    """Test image preprocessing pipelines and inverses."""

    def test_train_transforms_output_shape_and_type(self):
        pil_img = Image.new("RGB", (32, 32), color=(128, 128, 128))
        train_tf = get_train_transforms(augment=True)
        tensor = train_tf(pil_img)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 32, 32)
        assert tensor.dtype == torch.float32

    def test_test_transforms_deterministic(self):
        pil_img = Image.new("RGB", (32, 32), color=(200, 100, 50))
        test_tf = get_test_transforms()
        t1 = test_tf(pil_img)
        t2 = test_tf(pil_img)

        assert torch.equal(t1, t2)
        assert t1.shape == (3, 32, 32)

    def test_inverse_transforms(self):
        pil_img = Image.new("RGB", (32, 32), color=(100, 150, 200))
        test_tf = get_test_transforms()
        inv_tf = get_inverse_transforms()

        norm_tensor = test_tf(pil_img)
        recovered_tensor = inv_tf(norm_tensor)

        # Expected range [0, 1]
        assert recovered_tensor.min() >= -1e-5
        assert recovered_tensor.max() <= 1.0 + 1e-5


@pytest.mark.unit
class TestSplits:
    """Test stratified splitting and evaluation subset generation."""

    def test_stratified_split_counts_and_disjointness(self):
        # 50,000 mock targets (5,000 per class across 10 classes)
        mock_targets = [i % 10 for i in range(50000)]
        train_idx, val_idx = create_stratified_train_val_split(mock_targets, val_ratio=0.1, seed=42)

        assert len(train_idx) == 45000
        assert len(val_idx) == 5000
        assert len(set(train_idx).intersection(set(val_idx))) == 0

        # Validate class balance in validation set
        val_targets = [mock_targets[i] for i in val_idx]
        counts = np.bincount(val_targets, minlength=10)
        assert (counts == 500).all()

        validate_splits(train_idx, val_idx, total_expected=50000)

    def test_fixed_eval_indices(self):
        # 10,000 test targets (1,000 per class)
        mock_targets = [i % 10 for i in range(10000)]
        eval_idx = create_fixed_eval_indices(mock_targets, num_samples=1000, seed=42)

        assert len(eval_idx) == 1000
        assert len(set(eval_idx)) == 1000

        eval_targets = [mock_targets[i] for i in eval_idx]
        counts = np.bincount(eval_targets, minlength=10)
        assert (counts == 100).all()


@pytest.mark.unit
class TestValidation:
    """Test batch and split validation functions."""

    def test_validate_batch_success(self):
        images = torch.randn(16, 3, 32, 32)
        labels = torch.randint(0, 10, (16,))
        # Should execute without error
        validate_batch(images, labels, num_classes=10)

    def test_validate_batch_invalid_shape_raises(self):
        images_3d = torch.randn(3, 32, 32)
        labels = torch.randint(0, 10, (16,))
        with pytest.raises(DatasetError):
            validate_batch(images_3d, labels)

    def test_validate_batch_nan_raises(self):
        images = torch.randn(16, 3, 32, 32)
        images[0, 0, 0, 0] = float("nan")
        labels = torch.randint(0, 10, (16,))
        with pytest.raises(DatasetError):
            validate_batch(images, labels)

    def test_validate_batch_out_of_bounds_label_raises(self):
        images = torch.randn(16, 3, 32, 32)
        labels = torch.tensor([0, 1, 15, 2] + [0] * 12)
        with pytest.raises(DatasetError):
            validate_batch(images, labels, num_classes=10)

    def test_validate_splits_overlap_raises(self):
        train_idx = [0, 1, 2, 3]
        val_idx = [3, 4, 5]
        with pytest.raises(DatasetError):
            validate_splits(train_idx, val_idx, total_expected=6)


@pytest.mark.unit
class TestDatasetMetadata:
    """Test canonical CIFAR-10 metadata descriptor."""

    def test_cifar10_metadata(self):
        meta = CIFAR10_METADATA
        assert meta.name == "cifar10"
        assert meta.num_classes == 10
        assert meta.image_shape == (3, 32, 32)
        assert len(meta.class_names) == 10
        assert meta.train_count + meta.val_count == 50000
