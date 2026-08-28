"""VulnShield-DNN Reporting and Publication Artifact Package."""

from vulnshield.reporting.tables import (
    generate_baseline_comparison_table,
    generate_protection_budget_table
)
from vulnshield.reporting.figures import (
    plot_discovery_comparison,
    plot_budget_tradeoff_curve,
    plot_radar_evaluation
)
from vulnshield.reporting.generator import build_research_report

__all__ = [
    "generate_baseline_comparison_table",
    "generate_protection_budget_table",
    "plot_discovery_comparison",
    "plot_budget_tradeoff_curve",
    "plot_radar_evaluation",
    "build_research_report"
]
