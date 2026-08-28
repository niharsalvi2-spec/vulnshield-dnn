"""Publication Figure Generation Utilities (Bar Charts, Curves, Radar Plots)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_discovery_comparison(
    data: List[Dict[str, Any]],
    output_path: Union[str, Path],
    title: str = "Comparison of Vulnerability Discovery Methods"
) -> None:
    """Generate a publication-quality bar chart comparing discovery methods."""
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    methods = [d["method"] for d in data]
    top_deltas = [d.get("top_delta", 0.0) for d in data]
    mean_deltas = [d.get("mean_delta", 0.0) for d in data]

    x = np.arange(len(methods))
    width = 0.35

    plt.figure(figsize=(9, 5.5))
    sns.set_theme(style="whitegrid")

    plt.bar(x - width/2, top_deltas, width, label="Max $\\Delta A$ Discovered (%)", color="#d9534f")
    plt.bar(x + width/2, mean_deltas, width, label="Mean $\\Delta A$ Sampled (%)", color="#337ab7")

    plt.xlabel("Discovery Method", fontweight="bold", labelpad=8)
    plt.ylabel("Accuracy Degradation $\\Delta A$ (%)", fontweight="bold", labelpad=8)
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xticks(x, methods)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(str(out_file), dpi=300)
    plt.close()


def plot_budget_tradeoff_curve(
    budgets: Sequence[float],
    clean_accuracies: Sequence[float],
    fault_accuracies: Sequence[float],
    output_path: Union[str, Path],
    title: str = "Protection Budget Trade-Off Curve"
) -> None:
    """Generate trade-off curve across protection budgets (0%, 1%, 3%, 5%, 10%)."""
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    pct_labels = [f"{int(b * 100)}%" for b in budgets]

    plt.figure(figsize=(8, 5))
    sns.set_theme(style="whitegrid")

    plt.plot(pct_labels, clean_accuracies, marker="o", linewidth=2.2, label="Clean Test Accuracy (%)", color="#2e6da4")
    plt.plot(pct_labels, fault_accuracies, marker="s", linewidth=2.2, label="Fault Accuracy under Attack (%)", color="#4cae4c")

    plt.xlabel("Channel Protection Budget (%)", fontweight="bold", labelpad=8)
    plt.ylabel("Accuracy (%)", fontweight="bold", labelpad=8)
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(str(out_file), dpi=300)
    plt.close()


def plot_radar_evaluation(
    categories: Sequence[str],
    baseline_scores: Sequence[float],
    protected_scores: Sequence[float],
    output_path: Union[str, Path],
    title: str = "Multi-Dimensional Robustness Comparison"
) -> None:
    """Generate a spider/radar chart comparing clean baseline vs protected model."""
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    n_cats = len(categories)
    angles = [n / float(n_cats) * 2 * math.pi for n in range(n_cats)]
    angles += angles[:1]

    b_vals = list(baseline_scores) + [baseline_scores[0]]
    p_vals = list(protected_scores) + [protected_scores[0]]

    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)

    plt.xticks(angles[:-1], categories, size=10, fontweight="bold")
    ax.set_rlabel_position(0)
    plt.yticks([20, 40, 60, 80, 100], ["20%", "40%", "60%", "80%", "100%"], color="grey", size=8)
    plt.ylim(0, 100)

    # Baseline curve
    ax.plot(angles, b_vals, linewidth=2, linestyle="solid", label="Clean Baseline", color="#d9534f")
    ax.fill(angles, b_vals, color="#d9534f", alpha=0.15)

    # Protected curve
    ax.plot(angles, p_vals, linewidth=2, linestyle="solid", label="VulnShield Protected (5%)", color="#2e6da4")
    ax.fill(angles, p_vals, color="#2e6da4", alpha=0.25)

    plt.title(title, size=13, fontweight="bold", y=1.08)
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    plt.savefig(str(out_file), dpi=300)
    plt.close()
