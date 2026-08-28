"""One-Command CLI Entry Point to Run the Full VulnShield-DNN Research Pipeline."""

import argparse
from pathlib import Path

from vulnshield.pipeline import PipelineConfig, VulnShieldMasterPipeline


def main():
    parser = argparse.ArgumentParser(description="Run complete VulnShield-DNN end-to-end pipeline.")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "vgg16"], help="Target architecture")
    parser.add_argument("--seed", type=int, default=42, help="Reproducibility seed")
    parser.add_argument("--train-epochs", type=int, default=100, help="Clean training epochs")
    parser.add_argument("--discovery-episodes", type=int, default=20, help="TD3 discovery episodes")
    parser.add_argument("--fine-tune-epochs", type=int, default=30, help="Fault-aware fine-tuning epochs")
    parser.add_argument("--device", type=str, default=None, help="Compute device (auto, cuda, cpu)")
    args = parser.parse_args()

    cfg = PipelineConfig(
        model_name=args.model,
        seed=args.seed,
        train_epochs=args.train_epochs,
        discovery_episodes=args.discovery_episodes,
        fine_tune_epochs=args.fine_tune_epochs,
        device=args.device
    )

    repo_root = Path(__file__).resolve().parent.parent.parent
    pipeline = VulnShieldMasterPipeline(config=cfg, project_root=repo_root)
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
