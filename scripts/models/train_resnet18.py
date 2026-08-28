"""CLI Script to Train Clean Baseline ResNet-18 on CIFAR-10."""

import argparse
from pathlib import Path

from vulnshield.utils.config import load_yaml, resolve_project_paths
from vulnshield.utils.reproducibility import set_seed
from vulnshield.utils.device import get_device
from vulnshield.data.loaders import build_cifar10_dataloaders
from vulnshield.models.model_factory import create_model
from vulnshield.training.trainer import BaseTrainer, TrainerConfig
from vulnshield.training.evaluator import evaluate_model


def train_resnet18(epochs: int = 200, batch_size: int = 128, lr: float = 0.1, seed: int = 42):
    repo_root = Path(__file__).resolve().parent.parent.parent
    paths_cfg = load_yaml(repo_root / "configs/project/paths.yaml")
    resolved_paths = resolve_project_paths(paths_cfg, project_root=repo_root)

    data_dir = Path(resolved_paths.paths.data.raw)
    ckpt_dir = Path(resolved_paths.paths.checkpoints.base_models) / "resnet18"

    set_seed(seed)
    device = get_device()

    print("=" * 65)
    print("      VulnShield-DNN: Train Clean ResNet-18 on CIFAR-10")
    print("=" * 65)
    print(f"[*] Compute Device : {device}")
    print(f"[*] Dataset Dir    : {data_dir}")
    print(f"[*] Checkpoint Dir : {ckpt_dir}")
    print(f"[*] Epochs         : {epochs}")
    print(f"[*] Batch Size     : {batch_size}")
    print(f"[*] Learning Rate  : {lr}")
    print(f"[*] Seed           : {seed}\n")

    # 1. Load Data
    cifar10_cfg = load_yaml(repo_root / "configs/data/cifar10.yaml")
    splits_cfg = load_yaml(repo_root / "configs/data/dataset_splits.yaml")
    merged_cfg = {**cifar10_cfg, **splits_cfg}
    merged_cfg["dataloader"]["train_batch_size"] = batch_size

    loaders = build_cifar10_dataloaders(data_dir=data_dir, config=merged_cfg, seed=seed)

    # 2. Instantiate Model
    model = create_model("resnet18", num_classes=10, device=device)

    # 3. Configure Trainer
    trainer_cfg = TrainerConfig(
        epochs=epochs,
        learning_rate=lr,
        optimizer_name="sgd",
        momentum=0.9,
        weight_decay=5e-4,
        scheduler_name="cosine",
        eta_min=1e-5
    )
    trainer = BaseTrainer(model=model, config=trainer_cfg, device=device)

    # 4. Train
    results = trainer.fit(
        train_loader=loaders.train,
        val_loader=loaders.val,
        checkpoint_dir=ckpt_dir,
        checkpoint_name="resnet18_clean"
    )

    # 5. Evaluate on Test Set
    test_res = evaluate_model(model, loaders.test, device=device)
    print(f"\n[FINAL CLEAN TEST EVALUATION] ResNet-18 Clean Test Accuracy: {test_res.accuracy:.2f}% (Loss: {test_res.loss:.4f})")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Train Clean ResNet-18 on CIFAR-10")
    parser.add_argument("--epochs", type=int, default=200, help="Total training epochs (default: 200)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_resnet18(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed)


if __name__ == "__main__":
    main()
