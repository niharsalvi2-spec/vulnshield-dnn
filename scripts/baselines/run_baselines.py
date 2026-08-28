"""CLI Script to Execute Discovery Baselines (Random, Activation, Taylor/Gradient, DDPG)."""

import argparse
import json
from pathlib import Path
import torch

from vulnshield.utils.config import load_yaml, resolve_project_paths
from vulnshield.utils.device import get_device
from vulnshield.data.loaders import build_cifar10_dataloaders
from vulnshield.models.model_factory import create_model, load_model_weights
from vulnshield.training.evaluator import evaluate_model
from vulnshield.baselines import (
    run_random_baseline,
    run_activation_baseline,
    run_gradient_baseline,
    DDPGAgent,
    DDPGConfig
)
from vulnshield.discovery.env import FaultDiscoveryEnv


def run_all_baselines(
    model_name: str,
    checkpoint_path: str,
    budget: int = 50,
    seed: int = 42
):
    repo_root = Path(__file__).resolve().parent.parent.parent
    paths_cfg = load_yaml(repo_root / "configs/project/paths.yaml")
    resolved_paths = resolve_project_paths(paths_cfg, project_root=repo_root)
    data_dir = Path(resolved_paths.paths.data.raw)
    out_dir = Path(resolved_paths.paths.results.discovery) / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()

    print("=" * 65)
    print("      VulnShield-DNN: Discovery Baselines Execution")
    print("=" * 65)
    print(f"[*] Model        : {model_name}")
    print(f"[*] Checkpoint   : {checkpoint_path}")
    print(f"[*] Budget       : {budget} evaluations")
    print(f"[*] Output Dir   : {out_dir}")
    print(f"[*] Device       : {device}\n")

    cifar10_cfg = load_yaml(repo_root / "configs/data/cifar10.yaml")
    splits_cfg = load_yaml(repo_root / "configs/data/dataset_splits.yaml")
    merged_cfg = {**cifar10_cfg, **splits_cfg}

    loaders = build_cifar10_dataloaders(data_dir=data_dir, config=merged_cfg, seed=seed)
    model = create_model(model_name, num_classes=10, device=device)
    load_model_weights(model, checkpoint_path, device=device)

    # 1. Clean accuracy
    clean_eval = evaluate_model(model, loaders.eval_fault, device=device)
    clean_acc = clean_eval.accuracy
    print(f"[*] Baseline Clean Accuracy (eval set): {clean_acc:.2f}%\n")

    # 2. Random Baseline
    print("[1/4] Running Random Baseline...")
    random_res = run_random_baseline(
        model=model,
        dataloader=loaders.eval_fault,
        clean_accuracy=clean_acc,
        budget=budget,
        seed=seed,
        device=device
    )
    with open(out_dir / f"{model_name}_random_baseline.json", "w") as f:
        json.dump(random_res, f, indent=2)
    print(f"  -> Top ΔA found by Random: {random_res[0]['delta_accuracy']:.2f}%")

    # 3. Activation Magnitude Baseline
    print("\n[2/4] Running Activation Magnitude Baseline...")
    act_res = run_activation_baseline(
        model=model,
        dataloader=loaders.eval_fault,
        budget=budget,
        device=device
    )
    with open(out_dir / f"{model_name}_activation_baseline.json", "w") as f:
        json.dump(act_res, f, indent=2)
    print(f"  -> Top Activation Score: {act_res[0]['activation_score']:.4f}")

    # 4. Gradient / Taylor Baseline
    print("\n[3/4] Running Taylor / Gradient Sensitivity Baseline...")
    grad_res = run_gradient_baseline(
        model=model,
        dataloader=loaders.eval_fault,
        budget=budget,
        device=device
    )
    with open(out_dir / f"{model_name}_gradient_baseline.json", "w") as f:
        json.dump(grad_res, f, indent=2)
    print(f"  -> Top Gradient Score: {grad_res[0]['gradient_score']:.6f}")

    # 5. Layer-wise DDPG Baseline
    print("\n[4/4] Running Layer-wise DDPG Agent Baseline...")
    env = FaultDiscoveryEnv(
        model=model,
        dataloader=loaders.eval_fault,
        clean_accuracy=clean_acc,
        budget=budget,
        device=device
    )
    ddpg_agent = DDPGAgent(obs_dim=env.obs_dim, action_dim=env.action_dim, device=device)
    ddpg_res = ddpg_agent.run_discovery(env, num_episodes=10)
    with open(out_dir / f"{model_name}_ddpg_baseline.json", "w") as f:
        json.dump(ddpg_res, f, indent=2)

    print(f"\n[PASS] All baselines completed. Results saved to: {out_dir}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Run discovery baselines on a trained model.")
    parser.add_argument("--model", type=str, required=True, choices=["resnet18", "vgg16"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_all_baselines(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        budget=args.budget,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
