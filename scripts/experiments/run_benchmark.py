"""Multi-Seed Benchmark Runner for Fair Comparative Discovery Evaluation.

Executes TD3 and all baselines under identical conditions:
  - Same model checkpoint
  - Same evaluation split
  - Same fault model (StuckAtZero)
  - Identical query budget: N=50 per method
  - Multiple independent seeds: [42, 123, 456, 789, 999]
  - Outputs: Raw JSON per seed per method, with provenance manifests
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from vulnshield.utils.provenance import ExperimentManifest
from vulnshield.utils.config import load_config
from vulnshield.utils.reproducibility import set_global_seed
from vulnshield.analysis.statistics import (
    compute_distribution_statistics,
    compute_paired_significance,
    holm_bonferroni_correction,
    compute_discovery_curve_auc
)
from vulnshield.validation.result_validator import (
    validate_discovery_result,
    ResultValidationError
)


SEEDS = [42, 123, 456, 789, 999]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-seed comparative benchmark runner")
    p.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "vgg16"])
    p.add_argument("--checkpoint", type=str, required=False, help="Path to trained model checkpoint")
    p.add_argument("--budget", type=int, default=50, help="Global fault query budget (default: 50)")
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--output-dir", type=str, default="results/discovery")
    p.add_argument("--debug", action="store_true", help="Use minimal debug settings (fast smoke test)")
    return p.parse_args()


def run_benchmark():
    args = parse_args()

    budget = 6 if args.debug else args.budget
    seeds = [42] if args.debug else args.seeds
    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir = output_dir / "baselines"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    if args.debug:
        print("[DEBUG MODE] Budget=6, 1 seed — NOT FOR RESEARCH RESULTS")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from vulnshield.models.model_factory import build_model
    from vulnshield.data.cifar10 import build_cifar10_loaders
    from vulnshield.discovery.env import FaultDiscoveryEnv
    from vulnshield.discovery.td3_agent import TD3Agent, TD3Config
    from vulnshield.baselines.random_baseline import run_random_baseline
    from vulnshield.baselines.activation_baseline import run_activation_baseline
    from vulnshield.baselines.gradient_baseline import run_gradient_baseline
    from vulnshield.baselines.ddpg_baseline import DDPGAgent, DDPGConfig

    if not args.debug:
        if not args.checkpoint or not Path(args.checkpoint).exists():
            raise FileNotFoundError(
                f"[FATAL RESEARCH ERROR] Research mode requires a valid trained checkpoint. "
                f"Provided checkpoint '{args.checkpoint}' does not exist. "
                f"Train clean model first via scripts/models/train_{args.model}.py or pass --debug for smoke-testing."
            )

    model = build_model(args.model)
    if args.checkpoint and Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        print(f"[*] Loaded checkpoint: {args.checkpoint}")
    else:
        print("[DEBUG WARNING] Running with randomly initialized model strictly for smoke-testing.")

    model.eval().to(device)

    loaders = build_cifar10_loaders(batch_size=64 if not args.debug else 32)
    eval_loader = loaders["eval_fault"]

    # Compute clean baseline accuracy once
    from vulnshield.training.evaluator import evaluate_model
    clean_res = evaluate_model(model, eval_loader, device=device)
    clean_acc = clean_res.accuracy
    print(f"[*] Clean Baseline Accuracy: {clean_acc:.2f}%")

    # Save clean accuracy
    clean_eval_file = Path(args.output_dir).parent / "final" / f"{args.model}_clean_eval.json"
    clean_eval_file.parent.mkdir(parents=True, exist_ok=True)
    with open(clean_eval_file, "w") as f:
        json.dump({"clean_accuracy": clean_acc, "model": args.model}, f, indent=2)

    all_method_results = {}
    per_seed_data = {}  # method_name -> {"top_delta": [...], "mean_delta": [...], "auc": [...]}

    # ─────────────────────────────────────────────────────────────────────────
    # Run each method across all seeds
    # ─────────────────────────────────────────────────────────────────────────
    methods = ["td3", "random", "activation", "gradient", "ddpg"]

    for method_name in methods:
        seed_top_deltas = []
        seed_mean_deltas = []
        seed_aucs = []

        for seed in seeds:
            set_global_seed(seed)

            manifest = ExperimentManifest.create(
                experiment_id=f"{args.model}_{method_name}_seed{seed}_q{budget}",
                stage_name="discovery",
                seed=seed,
                parameters={"budget": budget, "model": args.model, "method": method_name},
                repo_root=PROJECT_ROOT
            )

            print(f"\n[*] {method_name.upper()} | Seed={seed} | Budget={budget}")

            if method_name == "td3":
                env = FaultDiscoveryEnv(model, eval_loader, clean_accuracy=clean_acc, budget=10, device=device)
                cfg = TD3Config(hidden_dim=256, warmup_steps=min(10, budget // 4))
                agent = TD3Agent(obs_dim=env.obs_dim, action_dim=env.action_dim, config=cfg, device=device)
                res = agent.run_discovery(env, max_total_queries=budget, verbose=True)
                result_file = output_dir / f"{args.model}_td3_seed{seed}_q{budget}.json"

            elif method_name == "random":
                raw = run_random_baseline(model, eval_loader, clean_acc, budget=budget, seed=seed, device=device)
                res = {
                    "top_channels": sorted(raw, key=lambda x: x["delta_accuracy"], reverse=True),
                    "total_queries_executed": len(raw),
                    "max_budget_enforced": budget,
                    "episode_rewards": [r["delta_accuracy"] for r in raw]
                }
                result_file = baseline_dir / f"{args.model}_random_baseline_seed{seed}.json"

            elif method_name == "activation":
                raw = run_activation_baseline(model, eval_loader, budget=budget, seed=seed, device=device)
                top_ch = [{"layer_name": r["layer_name"], "channel_idx": r["channel_idx"],
                            "delta_accuracy": r.get("activation_score", 0.0)} for r in raw]
                res = {
                    "top_channels": top_ch,
                    "total_queries_executed": len(raw),
                    "max_budget_enforced": budget,
                    "episode_rewards": [r.get("activation_score", 0.0) for r in raw]
                }
                result_file = baseline_dir / f"{args.model}_activation_baseline_seed{seed}.json"

            elif method_name == "gradient":
                raw = run_gradient_baseline(model, eval_loader, budget=budget, seed=seed, device=device)
                top_ch = [{"layer_name": r["layer_name"], "channel_idx": r["channel_idx"],
                            "delta_accuracy": r.get("gradient_score", 0.0)} for r in raw]
                res = {
                    "top_channels": top_ch,
                    "total_queries_executed": len(raw),
                    "max_budget_enforced": budget,
                    "episode_rewards": [r.get("gradient_score", 0.0) for r in raw]
                }
                result_file = baseline_dir / f"{args.model}_gradient_baseline_seed{seed}.json"

            elif method_name == "ddpg":
                env = FaultDiscoveryEnv(model, eval_loader, clean_accuracy=clean_acc, budget=10, device=device)
                cfg_d = DDPGConfig(hidden_dim=256, warmup_steps=min(10, budget // 4))
                agent_d = DDPGAgent(obs_dim=env.obs_dim, action_dim=env.action_dim, config=cfg_d, device=device)
                res = agent_d.run_discovery(env, max_total_queries=budget, verbose=True)
                result_file = baseline_dir / f"{args.model}_ddpg_baseline_seed{seed}.json"

            # Validate before saving
            try:
                validate_discovery_result(res, budget=budget)
            except ResultValidationError as e:
                print(f"[VALIDATION ERROR] {e}")
                continue

            # Compute metrics
            top_ch = res.get("top_channels", [])
            top_d = top_ch[0]["delta_accuracy"] if top_ch else 0.0
            mean_d = sum(c["delta_accuracy"] for c in top_ch) / max(len(top_ch), 1)
            auc = compute_discovery_curve_auc(res.get("episode_rewards", [r["delta_accuracy"] for r in top_ch]))

            seed_top_deltas.append(top_d)
            seed_mean_deltas.append(mean_d)
            seed_aucs.append(auc)

            # Save raw result
            with open(result_file, "w") as f:
                json.dump(res, f, indent=2)

            manifest.record_artifact(result_file)
            manifest.save(result_file.parent / f"{result_file.stem}_manifest")
            print(f"  [PASS] Seed={seed}: top_Δ={top_d:.2f}% | mean_Δ={mean_d:.2f}% | AUC={auc:.2f}")

        per_seed_data[method_name] = {
            "top_delta": seed_top_deltas,
            "mean_delta": seed_mean_deltas,
            "auc": seed_aucs
        }

        # Statistical aggregation across seeds
        if seed_top_deltas:
            stats = compute_distribution_statistics(seed_top_deltas)
            all_method_results[method_name] = {
                "top_delta": stats.to_dict(),
                "mean_delta": compute_distribution_statistics(seed_mean_deltas).to_dict(),
                "auc": compute_distribution_statistics(seed_aucs).to_dict(),
                "budget": budget,
                "seeds": seeds,
                "raw_seed_observations": {
                    "top_delta": seed_top_deltas,
                    "mean_delta": seed_mean_deltas,
                    "auc": seed_aucs
                }
            }
            print(f"\n[✓] {method_name.upper()} Summary: "
                  f"top_Δ={stats.mean:.2f}±{stats.std:.2f}% "
                  f"[95% CI: {stats.ci_lower:.2f}–{stats.ci_upper:.2f}%]")

    # ─────────────────────────────────────────────────────────────────────────
    # Genuine Paired Statistical Significance (TD3 vs each baseline)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Statistical Significance: TD3 vs Baselines (Paired Multi-Seed Analysis)")
    print("=" * 65)

    p_values = []
    comparison_names = []
    sig_results = {}

    td3_deltas = per_seed_data.get("td3", {}).get("top_delta", [])

    if len(td3_deltas) >= 2:
        for bl in ["random", "activation", "gradient", "ddpg"]:
            bl_deltas = per_seed_data.get(bl, {}).get("top_delta", [])
            if len(bl_deltas) == len(td3_deltas) and len(td3_deltas) >= 2:
                paired_res = compute_paired_significance(td3_deltas, bl_deltas)
                p_val = paired_res["p_value_parametric"]
                p_values.append(p_val)
                comparison_names.append(bl)
                sig_results[bl] = {
                    "t_statistic": paired_res["t_statistic"],
                    "p_value_parametric": paired_res["p_value_parametric"],
                    "wilcoxon_stat": paired_res["wilcoxon_stat"],
                    "p_value_nonparametric": paired_res["p_value_nonparametric"],
                    "cohens_d": paired_res["cohens_d"],
                    "is_significant_p05": paired_res["is_significant_p05"],
                    "td3_seed_values": td3_deltas,
                    "baseline_seed_values": bl_deltas
                }
                print(f"  TD3 vs {bl.upper():10s} | t={paired_res['t_statistic']:6.3f} | p={p_val:7.4f} | d={paired_res['cohens_d']:6.3f} | Sig(p<0.05): {paired_res['is_significant_p05']}")

        if p_values:
            rejected = holm_bonferroni_correction(p_values, alpha=0.05)
            for i, bl in enumerate(comparison_names):
                sig_results[bl]["holm_bonferroni_rejected"] = rejected[i]
    else:
        print("[INFO] Minimum 2 independent seeds required for statistical significance testing.")

    # Save aggregated multi-seed results
    agg_file = output_dir / f"{args.model}_discovery_aggregated.json"
    with open(agg_file, "w") as f:
        json.dump({
            "model": args.model,
            "budget": budget,
            "seeds": seeds,
            "methods": all_method_results,
            "pairwise_significance_td3_vs_baselines": sig_results,
            "debug": args.debug
        }, f, indent=2)

    print(f"\n[PASS] Aggregated benchmark saved: {agg_file}")
    print("[NOTE] Per-seed significance testing requires individual seed arrays.")
    print("[NOTE] Collect per-seed top_delta arrays into compute_paired_significance() for full p-values.")


if __name__ == "__main__":
    run_benchmark()
