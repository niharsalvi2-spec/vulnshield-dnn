"""Download CIFAR-10 using torchvision's built-in verified downloader.

torchvision.datasets.CIFAR10 handles:
  - Resumable download
  - MD5 checksum verification
  - Automatic extraction
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torchvision
import torch
from torch.utils.data import DataLoader

from vulnshield.utils.config import load_yaml, resolve_project_paths
from vulnshield.data.transforms import get_train_transforms, get_test_transforms
from vulnshield.data.validation import validate_batch


def main():
    repo_root = Path(__file__).resolve().parent.parent.parent
    paths_cfg = load_yaml(repo_root / "configs/project/paths.yaml")
    resolved = resolve_project_paths(paths_cfg, project_root=repo_root)
    raw_dir = Path(resolved.paths.data.raw)
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("    VulnShield-DNN: CIFAR-10 Download (torchvision)")
    print("=" * 65)
    print(f"[*] Target directory: {raw_dir}")
    print(f"[*] Using torchvision {torchvision.__version__} with MD5 checksum verification")
    print(f"[*] Downloading CIFAR-10 (170 MB from cs.toronto.edu)...\n")

    # torchvision handles checksums + extraction automatically
    train_ds = torchvision.datasets.CIFAR10(
        root=str(raw_dir), train=True, download=True, transform=None
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=str(raw_dir), train=False, download=True, transform=None
    )

    print(f"\n[PASS] CIFAR-10 ready:")
    print(f"  - Train samples : {len(train_ds):,}")
    print(f"  - Test  samples : {len(test_ds):,}")
    print(f"  - Classes       : {train_ds.classes}")

    # Quick batch validation
    print("\n[*] Validating a sample batch through the transform pipeline...")
    test_transform = get_test_transforms()

    # Manually apply transform to first batch
    import torch
    images = torch.stack([test_transform(train_ds[i][0]) for i in range(128)])
    labels = torch.tensor([train_ds[i][1] for i in range(128)])

    validate_batch(images, labels, num_classes=10)
    print(f"[PASS] Batch shape: {images.shape}, Labels range: [{labels.min()}, {labels.max()}]")
    print(f"[PASS] Image tensor stats: mean={images.mean():.4f}, std={images.std():.4f}")
    print("\n[PASS] CIFAR-10 is fully downloaded and validated. Ready to train!")
    print("=" * 65)


if __name__ == "__main__":
    main()
