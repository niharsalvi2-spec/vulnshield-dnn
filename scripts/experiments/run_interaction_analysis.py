"""CLI Script to Run Pairwise Fault Interaction Analysis on Top Discovered Channels."""

import argparse
import json
from pathlib import Path
import torch

from vulnshield.utils.config import load_yaml, resolve_project_paths
from vulnshield.utils.device import get_device
from vulnshield.data.loaders import build_cifar10_dataloaders
from vulnshield.models.model_factory import create_model, load_model_weights
from vulnshield.training.evaluator import evaluate_model
from vulnshield.interaction import (
    evaluate_pairwise_interactions,
    summarize_interactions,
    build_interaction_matrix,
    plot_interaction_heatmap
)


def run_interaction_analysis(
    model_name: str,
    checkpoint_path: str,
    top_k: int = 10,
    seed: int = 42
):
    repo_root = Path(__file__).resolve().parent.parent.parent
    paths_cfg = load_yaml(repo_root / "configs/project/paths.yaml")
    resolved_paths = resolve_project_paths(paths_cfg, project_root=repo_root)
    data_dir = Path(resolved_paths.paths.data.raw)
    out_dir = Path(resolved_paths.paths.results.interaction)
    fig_dir = Path(resolved_paths.paths.artifacts.figures)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    print("=" * 65)
    print("      VulnShield-DNN: Multi-Channel Fault Interaction Analysis")
    print("=" * 65)
    print(f"[*] Model        : {model_name}")
    print(f"[*] Checkpoint   : {checkpoint_path}")
    print(f"[*] Top-K Pairs  : Top {top_k} channels")
    print(f"[*] Device       : {device}\n")

    cifar10_cfg = load_yaml(repo_root / "configs/data/cifar10.yaml")
    splits_cfg = load_yaml(repo_root / "configs/data/dataset_splits.yaml")
    merged_cfg = {**cifar10_cfg, **splits_cfg}

    loaders = build_cifar10_dataloaders(data_dir=data_dir, config=merged_cfg, seed=seed)
    model = create_model(model_name, num_classes=10, device=device)
    load_model_weights(model, checkpoint_path, device=device)

    clean_eval = evaluate_model(model, loaders.eval_fault, device=device)
    clean_acc = clean_eval.accuracy
    print(f"[*] Baseline Clean Accuracy: {clean_acc:.2f}%\n")

    # Load candidate channels from previous discovery or take initial spread
    discovery_file = Path(resolved_paths.paths.results.discovery) / f"{model_name}_td3_discovery.json"
    if discovery_file.exists():
        with open(discovery_file, "r") as f:
            disc_data = json.load(f)
            candidate_channels = [
                (d["layer_name"], d["channel_idx"]) for d in disc_data["top_channels"][:top_k]
            ]
    else:
        # Pick top spread of channels across layers
        from vulnshield.fault_injection.fault_injector import FaultInjector
        injector = FaultInjector(model)
        injectable = injector.list_injectable_layers()
        candidate_channels = [
            (layer_name, c) for layer_name, _ in injectable[:top_k] for c in [0]
        ][:top_k]

    print(f"[*] Analyzing pairwise interactions across {len(candidate_channels)} candidate channels...")
    results = evaluate_pairwise_interactions(
        model=model,
        channels=candidate_channels,
        dataloader=loaders.eval_fault,
        clean_accuracy=clean_acc,
        device=device
    )

    summary = summarize_interactions(results)
    print("\n[*] Interaction Summary:")
    print(f"  - Total Evaluated Pairs : {summary.total_pairs}")
    print(f"  - Synergistic (Compounding): {summary.num_synergistic} ({summary.num_synergistic/max(summary.total_pairs,1)*100:.1f}%)")
    print(f"  - Masking (Antagonistic)   : {summary.num_masking} ({summary.num_masking/max(summary.total_pairs,1)*100:.1f}%)")
    print(f"  - Additive (Independent)   : {summary.num_additive} ({summary.num_additive/max(summary.total_pairs,1)*100:.1f}%)")

    # Save JSON results
    out_json = out_dir / f"{model_name}_interaction_results.json"
    with open(out_json, "w") as f:
        json.dump({
            "summary": summary.to_dict(),
            "results": [r.to_dict() for r in results]
        }, f, indent=2)
    print(f"\n[PASS] Interaction results exported to: {out_json}")

    # Generate and save heatmap figure
    matrix, labels = build_interaction_matrix(results, candidate_channels)
    fig_path = fig_dir / f"{model_name}_interaction_heatmap.png"
    plot_interaction_heatmap(matrix, labels, output_path=fig_path, title=f"Interaction Matrix I(A,B) — {model_name.upper()}")
    print(f"[PASS] Interaction heatmap saved to: {fig_path}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Run multi-fault interaction analysis.")
    parser.add_argument("--model", type=str, required=True, choices=["resnet18", "vgg16"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_interaction_analysis(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        top_k=args.top_k,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
