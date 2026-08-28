"""Academic Table Generation in LaTeX and Markdown Formats."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def generate_baseline_comparison_table(
    data: List[Dict[str, Any]],
    output_format: str = "markdown"
) -> str:
    """Generate comparative evaluation table across discovery baselines.

    Args:
        data: List of dicts with keys: 'method', 'top_delta', 'mean_delta', 'budget'.
        output_format: 'markdown' or 'latex'.

    Returns:
        Formatted table string.
    """
    if output_format.lower() == "latex":
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Comparison of Channel Vulnerability Discovery Methods}",
            r"\label{tab:discovery_comparison}",
            r"\begin{tabular}{lcccc}",
            r"\hline",
            r"\textbf{Discovery Method} & \textbf{Top $\Delta A$ (\%)} & \textbf{Mean $\Delta A$ (\%)} & \textbf{Budget} & \textbf{Strategy} \\",
            r"\hline"
        ]
        for row in data:
            lines.append(
                f"{row.get('method', 'N/A')} & {row.get('top_delta', 0.0):.2f} & "
                f"{row.get('mean_delta', 0.0):.2f} & {row.get('budget', 50)} & "
                f"{row.get('strategy', 'Heuristic')} \\\\"
            )
        lines.extend([
            r"\hline",
            r"\end{tabular}",
            r"\end{table}"
        ])
        return "\n".join(lines)

    # Markdown format
    lines = [
        "| Discovery Method | Top $\\Delta A$ (%) | Mean $\\Delta A$ (%) | Query Budget | Strategy |",
        "|:---|:---:|:---:|:---:|:---:|"
    ]
    for row in data:
        lines.append(
            f"| **{row.get('method', 'N/A')}** | {row.get('top_delta', 0.0):.2f}% | "
            f"{row.get('mean_delta', 0.0):.2f}% | {row.get('budget', 50)} | "
            f"{row.get('strategy', 'Heuristic')} |"
        )
    return "\n".join(lines)


def generate_protection_budget_table(
    data: List[Dict[str, Any]],
    output_format: str = "markdown"
) -> str:
    """Generate multi-budget protection comparison table across all 4 budgets.

    Args:
        data: List of dicts with keys: 'model_label', 'clean_acc', 'known_acc',
              'unseen_acc', 'two_fault_acc', 'five_fault_acc'.
        output_format: 'markdown' or 'latex'.

    Returns:
        Formatted table string.
    """
    if output_format.lower() == "latex":
        lines = [
            r"\begin{table*}[htbp]",
            r"\centering",
            r"\caption{Multi-Dimensional Robustness across Protection Budgets (1\%, 3\%, 5\%, 10\%)}",
            r"\label{tab:protection_budgets}",
            r"\begin{tabular}{lccccc}",
            r"\hline",
            r"\textbf{Protection Setting} & \textbf{Clean Acc (\%)} & \textbf{Known Fault Acc (\%)} & \textbf{Unseen Fault Acc (\%)} & \textbf{2-Fault Acc (\%)} & \textbf{5-Fault Acc (\%)} \\",
            r"\hline"
        ]
        for row in data:
            lines.append(
                f"{row.get('model_label', 'N/A')} & {row.get('clean_acc', 0.0):.2f} & "
                f"{row.get('known_acc', 0.0):.2f} & {row.get('unseen_acc', 0.0):.2f} & "
                f"{row.get('two_fault_acc', 0.0):.2f} & {row.get('five_fault_acc', 0.0):.2f} \\\\"
            )
        lines.extend([
            r"\hline",
            r"\end{tabular}",
            r"\end{table*}"
        ])
        return "\n".join(lines)

    # Markdown format
    lines = [
        "| Protection Setting | Clean Acc (%) | Known Fault Acc (%) | Unseen Gen. Acc (%) | 2-Fault Acc (%) | 5-Fault Acc (%) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|"
    ]
    for row in data:
        lines.append(
            f"| **{row.get('model_label', 'N/A')}** | {row.get('clean_acc', 0.0):.2f}% | "
            f"{row.get('known_acc', 0.0):.2f}% | {row.get('unseen_acc', 0.0):.2f}% | "
            f"{row.get('two_fault_acc', 0.0):.2f}% | {row.get('five_fault_acc', 0.0):.2f}% |"
        )
    return "\n".join(lines)
