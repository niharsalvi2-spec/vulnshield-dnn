"""CLI Script to Execute the Complete 6-Dimensional Evaluation Suite on a Model Checkpoint."""

import argparse
import json
from pathlib import Path
import torch

from vulnshield.utils.config import load_yaml, resolve_project_paths
from vulnshield.utils.device import get_device
from vulnshield.data.loaders import build_cifar10_dataloaders
from vulnshield.models.model_factory import create_model, load_model_weights
from vulnshield.models.common import get_named_conv_layers
from vulnshield.evaluation import (
    ComprehensiveEvaluationReport,
    evaluate_clean_preservation,
    evaluate_channel_fault_set,
    evaluate_unseen_channel_generalization,
    evaluate_simultaneous_multi_faults,
    evaluate_bit_flip_robustness,
    evaluate_adversarial_robustness
)


def run_full_evaluation(
    model_name: str,
    checkpoint_path: str,
    baseline_clean_acc: float = 93.0,
    seed: int = 42
):
    repo_root = Path(__file__).resolve().parent.parent.parent
    paths_cfg = load_yaml(repo_root / "configs/project/paths.yaml")
    resolved_paths = resolve_project_paths(paths_cfg, project_root=repo_root)
    data_dir = Path(resolved_paths.paths.data.raw)
    out_dir = Path(resolved_paths.paths.results.final)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    print("=" * 70)
    print("      VulnShield-DNN: Comprehensive Multi-Dimensional Evaluation")
    print("=" * 70)
    print(f"[*] Model         : {model_name}")
    print(f"[*] Checkpoint    : {checkpoint_path}")
    print(f"[*] Baseline Acc  : {baseline_clean_acc:.2f}%")
    print(f"[*] Device        : {device}\n")

    cifar10_cfg = load_yaml(repo_root / "configs/data/cifar10.yaml")
    splits_cfg = load_yaml(repo_root / "configs/data/dataset_splits.yaml")
    merged_cfg = {**cifar10_cfg, **splits_cfg}
    loaders = build_cifar10_dataloaders(data_dir=data_dir, config=merged_cfg, seed=seed)

    model = create_model(model_name, num_classes=10, device=device)
    load_model_weights(model, checkpoint_path, device=device)

    from vulnshield.fault_injection.fault_injector import FaultInjector
    injector = FaultInjector(model)
    all_injectable = injector.list_injectable_layers()

    # 1. Clean Accuracy Preservation
    print("[1/6] Dimension 1: Clean Accuracy Preservation...")
    clean_res, clean_drop, pass_clean = evaluate_clean_preservation(
        model, loaders.test, baseline_clean_accuracy=baseline_clean_acc, device=device
    )
    print(f"  -> Clean Test Acc: {clean_res.accuracy:.2f}% (Drop: {clean_drop:+.2f}%, Within <=1% tolerance: {pass_clean})")

    # 2. Known Channel Faults
    print("\n[2/6] Dimension 2: Known Protected Channel Fault Accuracy...")
    known_channels = [
        (name, c) for name, layer in get_named_conv_layers(model)[:5] for c in range(min(2, layer.out_channels))
    ]
    known_acc, known_drop, _ = evaluate_channel_fault_set(
        model, known_channels, loaders.eval_fault, clean_accuracy=clean_res.accuracy, device=device
    )
    print(f"  -> Mean Known Fault Acc: {known_acc:.2f}% (Drop: {known_drop:.2f}%)")

    # 3. Unseen Channel Fault Generalization
    print("\n[3/6] Dimension 3: Unseen Channel Fault Generalization...")
    unseen_acc, unseen_drop, _ = evaluate_unseen_channel_generalization(
        model, known_channels, all_injectable, loaders.eval_fault,
        clean_accuracy=clean_res.accuracy, num_unseen_samples=20, seed=seed, device=device
    )
    print(f"  -> Mean Unseen Fault Acc: {unseen_acc:.2f}% (Drop: {unseen_drop:.2f}%)")

    # 4. Simultaneous Multi-Fault Stress Test
    print("\n[4/6] Dimension 4: Simultaneous Multi-Fault Robustness (2, 3, 5 channels)...")
    multi_res = evaluate_simultaneous_multi_faults(
        model, all_injectable, loaders.eval_fault, fault_counts=(2, 3, 5), trials_per_count=10, seed=seed, device=device
    )
    for k, acc in multi_res.items():
        print(f"  -> {k} Simultaneous Faults Acc: {acc:.2f}%")

    # 5. Physical Bit-Flip Simulation
    print("\n[5/6] Dimension 5: Physical Bit-Flip Fault Simulation (IEEE 754 float32)...")
    bit_res = evaluate_bit_flip_robustness(
        model, loaders.eval_fault, target_bits=("sign", "exponent", "mantissa"), flips_per_layer=5, seed=seed, device=device
    )
    for b_type, acc in bit_res.items():
        print(f"  -> {b_type.capitalize()} Bit-Flip Acc: {acc:.2f}%")

    # 6. Decoupled Adversarial Comparison
    print("\n[6/6] Dimension 6: Decoupled Adversarial Robustness (FGSM & PGD-20)...")
    fgsm_acc, pgd_acc = evaluate_adversarial_robustness(
        model, loaders.eval_fault, fgsm_epsilon=8/255, pgd_epsilon=8/255, pgd_steps=20, device=device
    )
    print(f"  -> FGSM (eps=8/255) Acc: {fgsm_acc:.2f}%")
    print(f"  -> PGD-20 (eps=8/255) Acc: {pgd_acc:.2f}%")

    # Assemble Report
    report = ComprehensiveEvaluationReport(
        model_name=model_name,
        checkpoint_name=Path(checkpoint_path).stem,
        clean_accuracy=clean_res.accuracy,
        clean_loss=clean_res.loss,
        known_fault_accuracy=known_acc,
        known_fault_drop=known_drop,
        unseen_fault_accuracy=unseen_acc,
        unseen_fault_drop=unseen_drop,
        multi_fault_accuracies=multi_res,
        bit_flip_accuracies=bit_res,
        fgsm_accuracy=fgsm_acc,
        pgd_accuracy=pgd_acc
    )

    out_json = out_dir / f"{model_name}_full_evaluation_report.json"
    with open(out_json, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    print(f"\n[PASS] Full 6-Dimensional Evaluation Complete. Report saved to: {out_json}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive 6-dimensional evaluation.")
    parser.add_argument("--model", type=str, required=True, choices=["resnet18", "vgg16"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--baseline-acc", type=float, default=93.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_full_evaluation(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        baseline_clean_acc=args.baseline_acc,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
