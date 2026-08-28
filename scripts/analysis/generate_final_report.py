"""CLI Script to Generate Consolidated Final Research Report and Academic Tables."""

import argparse
import json
from pathlib import Path

from vulnshield.utils.config import load_yaml, resolve_project_paths
from vulnshield.reporting import (
    generate_baseline_comparison_table,
    generate_protection_budget_table,
    build_research_report
)


def generate_report(model_name: str):
    repo_root = Path(__file__).resolve().parent.parent.parent
    paths_cfg = load_yaml(repo_root / "configs/project/paths.yaml")
    resolved_paths = resolve_project_paths(paths_cfg, project_root=repo_root)

    reports_dir = Path(resolved_paths.paths.reports)
    tables_dir = Path(resolved_paths.paths.artifacts.tables)
    reports_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("      VulnShield-DNN: Automated Research Report Generation")
    print("=" * 65)
    print(f"[*] Target Model : {model_name}")
    print(f"[*] Reports Dir  : {reports_dir}")
    print(f"[*] Tables Dir   : {tables_dir}\n")

    # 1. Load Discovery Results
    disc_data = []
    discovery_dir = Path(resolved_paths.paths.results.discovery)
    td3_file = discovery_dir / f"{model_name}_td3_discovery.json"
    baselines_dir = discovery_dir / "baselines"

    if td3_file.exists():
        with open(td3_file, "r") as f:
            td3_res = json.load(f)
            top_ch = td3_res.get("top_channels", [])
            top_d = top_ch[0]["delta_accuracy"] if top_ch else 0.0
            mean_d = sum(c["delta_accuracy"] for c in top_ch) / max(len(top_ch), 1)
            disc_data.append({
                "method": "VulnShield TD3 (Ours)",
                "top_delta": top_d,
                "mean_delta": mean_d,
                "budget": td3_res.get("max_budget_enforced", 50),
                "strategy": "Twin Q + Policy Smoothing"
            })

    # Load baseline JSON files if present
    for b_name, b_strat in [
        ("random", "Uniform Random Sampling"),
        ("activation", "Mean L1 Activation Norm"),
        ("gradient", "Taylor 1st-Order Gradient"),
        ("ddpg", "Single Critic DDPG")
    ]:
        b_file = baselines_dir / f"{model_name}_{b_name}_baseline.json"
        if b_file.exists():
            with open(b_file, "r") as f:
                b_res = json.load(f)
                if isinstance(b_res, list) and b_res:
                    top_d = b_res[0].get("delta_accuracy", b_res[0].get("activation_score", 0.0))
                    mean_d = sum(r.get("delta_accuracy", 0.0) for r in b_res) / max(len(b_res), 1)
                    disc_data.append({
                        "method": b_name.capitalize(),
                        "top_delta": top_d,
                        "mean_delta": mean_d,
                        "budget": len(b_res),
                        "strategy": b_strat
                    })
                elif isinstance(b_res, dict) and "top_channels" in b_res:
                    top_ch = b_res["top_channels"]
                    top_d = top_ch[0]["delta_accuracy"] if top_ch else 0.0
                    mean_d = sum(c["delta_accuracy"] for c in top_ch) / max(len(top_ch), 1)
                    disc_data.append({
                        "method": b_name.upper(),
                        "top_delta": top_d,
                        "mean_delta": mean_d,
                        "budget": b_res.get("max_budget_enforced", 50),
                        "strategy": b_strat
                    })

    # 2. Load Protection Summaries
    prot_data = []
    prot_dir = Path(resolved_paths.paths.results.protection)
    for b_pct in [0.01, 0.03, 0.05, 0.10]:
        pct_int = int(b_pct * 100)
        p_file = prot_dir / f"{model_name}_protected_b{pct_int}pct_summary.json"
        if p_file.exists():
            with open(p_file, "r") as f:
                p_res = json.load(f)
                prot_data.append({
                    "model_label": f"Protected ({pct_int}% Budget)",
                    "clean_acc": p_res.get("best_clean_acc", 0.0),
                    "known_acc": p_res.get("best_fault_acc", 0.0),
                    "unseen_acc": p_res.get("unseen_gen_acc", 0.0),
                    "two_fault_acc": p_res.get("two_fault_acc", 0.0),
                    "five_fault_acc": p_res.get("five_fault_acc", 0.0)
                })

    # 3. Load Interaction Summary
    interaction_summary = None
    inter_file = Path(resolved_paths.paths.results.interaction) / f"{model_name}_interaction_results.json"
    if inter_file.exists():
        with open(inter_file, "r") as f:
            inter_res = json.load(f)
            interaction_summary = inter_res.get("summary", None)

    # Export LaTeX tables
    latex_disc = generate_baseline_comparison_table(disc_data, output_format="latex")
    latex_prot = generate_protection_budget_table(prot_data, output_format="latex")

    with open(tables_dir / f"{model_name}_table_discovery.tex", "w") as f:
        f.write(latex_disc)
    with open(tables_dir / f"{model_name}_table_protection.tex", "w") as f:
        f.write(latex_prot)

    print(f"[PASS] Exported LaTeX tables to {tables_dir}")

    # Export Markdown report
    out_md = reports_dir / f"{model_name}_final_research_report.md"
    build_research_report(
        model_name=model_name,
        baseline_discovery_data=disc_data,
        protection_budget_data=prot_data,
        output_path=out_md
    )
    print(f"[PASS] Final Research Report written to: {out_md}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Generate final report and tables.")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "vgg16"])
    args = parser.parse_args()

    generate_report(model_name=args.model)


if __name__ == "__main__":
    main()
