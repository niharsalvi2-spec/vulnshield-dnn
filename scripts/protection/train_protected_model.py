"""CLI Script to Execute Fault-Aware Fine-Tuning across Protection Budgets (1%, 3%, 5%, 10%)."""

import argparse
import json
from pathlib import Path
import torch

from vulnshield.utils.config import load_yaml, resolve_project_paths
from vulnshield.utils.reproducibility import set_seed
from vulnshield.utils.device import get_device
from vulnshield.data.loaders import build_cifar10_dataloaders
from vulnshield.models.model_factory import create_model, load_model_weights
from vulnshield.training.evaluator import evaluate_model
from vulnshield.protection import (
    calculate_budget_channel_count,
    select_top_k_channels,
    ProtectionBudget,
    FaultAwareTrainer,
    ProtectionTrainingConfig
)


def train_protected_model(
    model_name: str,
    baseline_checkpoint: str,
    budget_pct: float = 0.05,
    epochs: int = 30,
    lr: float = 0.01,
    seed: int = 42
):
    repo_root = Path(__file__).resolve().parent.parent.parent
    paths_cfg = load_yaml(repo_root / "configs/project/paths.yaml")
    resolved_paths = resolve_project_paths(paths_cfg, project_root=repo_root)
    data_dir = Path(resolved_paths.paths.data.raw)
    ckpt_dir = Path(resolved_paths.paths.checkpoints.protected) / model_name / f"b_{int(budget_pct*100)}pct"
    results_dir = Path(resolved_paths.paths.results.protection)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    device = get_device()

    print("=" * 65)
    print("      VulnShield-DNN: Fault-Aware Model Protection")
    print("=" * 65)
    print(f"[*] Model          : {model_name}")
    print(f"[*] Baseline Ckpt  : {baseline_checkpoint}")
    print(f"[*] Budget         : {budget_pct * 100:.1f}%")
    print(f"[*] Fine-tune Ep   : {epochs}")
    print(f"[*] Learning Rate  : {lr}")
    print(f"[*] Device         : {device}\n")

    cifar10_cfg = load_yaml(repo_root / "configs/data/cifar10.yaml")
    splits_cfg = load_yaml(repo_root / "configs/data/dataset_splits.yaml")
    merged_cfg = {**cifar10_cfg, **splits_cfg}
    loaders = build_cifar10_dataloaders(data_dir=data_dir, config=merged_cfg, seed=seed)

    # 1. Total network channels
    from vulnshield.models.common import get_named_conv_layers
    dummy = create_model(model_name, num_classes=10, device="cpu")
    conv_layers = get_named_conv_layers(dummy)
    total_network_channels = sum(layer.out_channels for _, layer in conv_layers)
    k_channels = calculate_budget_channel_count(total_network_channels, budget_pct)
    print(f"[*] Total Network Channels: {total_network_channels} -> Budget Channels: {k_channels}")

    # 2. Select protected channels from TD3 discovery results
    disc_file = Path(resolved_paths.paths.results.discovery) / f"{model_name}_td3_discovery.json"
    if disc_file.exists():
        with open(disc_file, "r") as f:
            disc_data = json.load(f)
            protected_channels = select_top_k_channels(disc_data["top_channels"], k_channels)
    else:
        # Initial layer spread if discovery file not found
        protected_channels = [
            (name, c) for name, layer in conv_layers for c in range(min(4, layer.out_channels))
        ][:k_channels]

    print(f"[*] Selected {len(protected_channels)} channels for fault-aware hardening.\n")

    # 3. Instantiate model and load baseline weights
    model = create_model(model_name, num_classes=10, device=device)
    load_model_weights(model, baseline_checkpoint, device=device)

    # 4. Configure and run FaultAwareTrainer
    prot_cfg = ProtectionTrainingConfig(
        epochs=epochs,
        learning_rate=lr,
        alpha=0.5,
        beta=0.5,
        lambda_drift=1e-4
    )
    trainer = FaultAwareTrainer(
        model=model,
        protected_channels=protected_channels,
        config=prot_cfg,
        device=device
    )

    results = trainer.fit(
        train_loader=loaders.train,
        val_loader=loaders.val,
        eval_fault_loader=loaders.eval_fault,
        checkpoint_dir=ckpt_dir,
        checkpoint_name=f"{model_name}_protected_b{int(budget_pct*100)}pct"
    )

    # Save summary
    out_json = results_dir / f"{model_name}_protected_b{int(budget_pct*100)}pct_summary.json"
    with open(out_json, "w") as f:
        json.dump({
            "model": model_name,
            "budget_pct": budget_pct,
            "num_protected_channels": len(protected_channels),
            "protected_channels": [f"{l}_c{c}" for l, c in protected_channels],
            "best_combined_score": results["best_combined_score"],
            "best_checkpoint": results["best_checkpoint"]
        }, f, indent=2)
    print(f"[PASS] Protection summary saved to: {out_json}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Train protected model with fault-aware fine-tuning.")
    parser.add_argument("--model", type=str, required=True, choices=["resnet18", "vgg16"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--budget", type=float, default=0.05, choices=[0.01, 0.03, 0.05, 0.10])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_protected_model(
        model_name=args.model,
        baseline_checkpoint=args.checkpoint,
        budget_pct=args.budget,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
