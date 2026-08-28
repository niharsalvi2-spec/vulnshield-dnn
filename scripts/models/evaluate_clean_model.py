"""CLI Script to Evaluate a Trained Model Checkpoint on Clean Test Data."""

import argparse
import json
from pathlib import Path
import torch

from vulnshield.utils.config import load_yaml, resolve_project_paths
from vulnshield.utils.device import get_device
from vulnshield.data.loaders import build_cifar10_dataloaders
from vulnshield.models.model_factory import create_model, load_model_weights
from vulnshield.training.evaluator import evaluate_model


def evaluate_clean_model(
    model_name: str,
    checkpoint_path: str,
    split: str = "test"
):
    repo_root = Path(__file__).resolve().parent.parent.parent
    paths_cfg = load_yaml(repo_root / "configs/project/paths.yaml")
    resolved_paths = resolve_project_paths(paths_cfg, project_root=repo_root)
    data_dir = Path(resolved_paths.paths.data.raw)
    device = get_device()

    print("=" * 65)
    print(f"      VulnShield-DNN: Clean Model Evaluation")
    print("=" * 65)
    print(f"[*] Model      : {model_name}")
    print(f"[*] Checkpoint : {checkpoint_path}")
    print(f"[*] Eval Split : {split}")
    print(f"[*] Device     : {device}\n")

    cifar10_cfg = load_yaml(repo_root / "configs/data/cifar10.yaml")
    splits_cfg = load_yaml(repo_root / "configs/data/dataset_splits.yaml")
    merged_cfg = {**cifar10_cfg, **splits_cfg}

    loaders = build_cifar10_dataloaders(data_dir=data_dir, config=merged_cfg, seed=42)

    model = create_model(model_name, num_classes=10, device=device)
    meta = load_model_weights(model, checkpoint_path, device=device)
    print(f"[PASS] Loaded checkpoint (trained epoch: {meta.get('epoch', 'N/A')}, val_acc: {meta.get('metrics', {}).get('val_acc', 'N/A')}%)")

    # Select dataloader
    if split == "test":
        loader = loaders.test
    elif split == "val":
        loader = loaders.val
    elif split == "train":
        loader = loaders.train
    else:
        loader = loaders.test

    result = evaluate_model(model, loader, device=device)
    result_dict = result.to_dict()
    result_dict["model"] = model_name
    result_dict["checkpoint"] = str(checkpoint_path)
    result_dict["split"] = split

    print(f"\n[*] Results on '{split}' split:")
    print(f"  - Clean Accuracy (Top-1) : {result.accuracy:.2f}%")
    print(f"  - Clean Accuracy (Top-5) : {result.top5_accuracy:.2f}%")
    print(f"  - Cross-Entropy Loss     : {result.loss:.4f}")
    print(f"  - Samples Evaluated      : {result.num_samples}")
    print(f"  - Evaluation Time        : {result.duration_seconds:.2f}s")
    print("=" * 65)

    # Save result
    results_dir = Path(resolved_paths.paths.results.evaluation)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"{model_name}_clean_eval_{split}.json"
    with open(out_file, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"[PASS] Evaluation result saved to: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained VulnShield-DNN model on clean CIFAR-10 data.")
    parser.add_argument("--model", type=str, required=True, choices=["resnet18", "vgg16"], help="Model architecture")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to evaluate on")
    args = parser.parse_args()

    evaluate_clean_model(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        split=args.split
    )


if __name__ == "__main__":
    main()
