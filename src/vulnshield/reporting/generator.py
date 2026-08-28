"""Automated Research Report Compiler."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from vulnshield.reporting.tables import (
    generate_baseline_comparison_table,
    generate_protection_budget_table
)


def build_research_report(
    model_name: str,
    baseline_discovery_data: List[Dict[str, Any]],
    protection_budget_data: List[Dict[str, Any]],
    interaction_summary: Optional[Dict[str, Any]] = None,
    output_path: Optional[Union[str, Path]] = None
) -> str:
    """Compile an academic final report in Markdown."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    doc = [
        f"# VulnShield-DNN: Research Findings & Empirical Evaluation Report",
        f"**Target Architecture:** {model_name.upper()} on CIFAR-10  ",
        f"**Generated:** {timestamp}  ",
        f"**Framework Version:** VulnShield-DNN v0.1.0  ",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "This report consolidates empirical experimental results evaluating:",
        "1. **Reinforcement Learning Discovery (TD3)** vs Heuristic Baselines (Random, Activation, Taylor/Gradient, DDPG).",
        "2. **Multi-Channel Fault Interactions** evaluating compounding failures $I(A, B) = E(A, B) - [E(A) + E(B)]$.",
        "3. **Fault-Aware Fine-Tuning Protection** under constrained budgets ($1\\%, 3\\%, 5\\%, 10\\%$).",
        "4. **Multi-Dimensional Robustness** across clean accuracy, unseen generalization, multi-fault stress, bit-flips, and adversarial attacks.",
        "",
        "---",
        "",
        "## 2. Vulnerability Discovery Benchmark",
        "Discovery methods evaluated under equal query budgets ($N=50$ evaluations):",
        "",
        generate_baseline_comparison_table(baseline_discovery_data, output_format="markdown"),
        "",
        "---",
        "",
        "## 3. Multi-Channel Fault Interaction Analysis",
    ]

    if interaction_summary:
        doc.extend([
            f"* **Total Evaluated Channel Pairs:** {interaction_summary.get('total_pairs', 'N/A')}",
            f"* **Synergistic (Compounding Failures):** {interaction_summary.get('num_synergistic', 0)} ({interaction_summary.get('pct_synergistic', 0)}%)",
            f"* **Masking Effects:** {interaction_summary.get('num_masking', 0)} ({interaction_summary.get('pct_masking', 0)}%)",
            f"* **Additive (Independent):** {interaction_summary.get('num_additive', 0)} ({interaction_summary.get('pct_additive', 0)}%)",
            "",
            "> [!IMPORTANT]",
            "> Channel faults cannot be treated independently. The presence of compounding failure pairs confirms the necessity of joint multi-hook evaluation.",
            ""
        ])
    else:
        doc.append("*Interaction analysis pending execution.*")

    doc.extend([
        "---",
        "",
        "## 4. Fault-Aware Protection Budget Evaluation",
        "Model robustness across the 4 formal channel protection allocations:",
        "",
        generate_protection_budget_table(protection_budget_data, output_format="markdown"),
        "",
        "---",
        "",
        "## 5. Artifact & Figure References",
        "- Discovery Comparison: `artifacts/figures/discovery_comparison.png`",
        "- Budget Trade-Off Curve: `artifacts/figures/budget_tradeoff_curve.png`",
        "- Interaction Heatmap: `artifacts/figures/interaction_heatmap.png`",
        "- Multi-Dimensional Radar Plot: `artifacts/figures/radar_evaluation.png`",
        "",
        "---",
        "*Report compiled automatically by VulnShield-DNN Reporting Engine.*"
    ])

    report_text = "\n".join(doc)

    if output_path is not None:
        out_file = Path(output_path).resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(report_text)

    return report_text
