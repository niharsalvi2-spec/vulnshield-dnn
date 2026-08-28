"""Unit Tests for Phase 11 — Reporting and Artifact Generation."""

from pathlib import Path
import pytest
import matplotlib
matplotlib.use("Agg")

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


@pytest.mark.unit
class TestTableGenerators:

    def test_baseline_comparison_markdown_table(self):
        data = [
            {"method": "Random", "top_delta": 5.0, "mean_delta": 1.0, "budget": 50, "strategy": "Uniform"},
            {"method": "TD3", "top_delta": 15.0, "mean_delta": 4.0, "budget": 50, "strategy": "RL"}
        ]
        md = generate_baseline_comparison_table(data, output_format="markdown")
        assert "| Discovery Method |" in md
        assert "| **Random** |" in md
        assert "| **TD3** |" in md

    def test_baseline_comparison_latex_table(self):
        data = [
            {"method": "Random", "top_delta": 5.0, "mean_delta": 1.0, "budget": 50, "strategy": "Uniform"},
            {"method": "TD3", "top_delta": 15.0, "mean_delta": 4.0, "budget": 50, "strategy": "RL"}
        ]
        tex = generate_baseline_comparison_table(data, output_format="latex")
        assert r"\begin{table}" in tex
        assert r"\end{table}" in tex
        assert "Random &" in tex

    def test_protection_budget_tables(self):
        data = [
            {"model_label": "Baseline", "clean_acc": 93.0, "known_acc": 75.0, "unseen_acc": 88.0, "two_fault_acc": 70.0, "five_fault_acc": 55.0},
            {"model_label": "Prot 5%", "clean_acc": 92.8, "known_acc": 90.0, "unseen_acc": 91.0, "two_fault_acc": 87.0, "five_fault_acc": 78.0}
        ]
        md = generate_protection_budget_table(data, output_format="markdown")
        tex = generate_protection_budget_table(data, output_format="latex")

        assert "| Protection Setting |" in md
        assert r"\begin{table*}" in tex


@pytest.mark.unit
class TestFigureGenerators:

    def test_plot_discovery_comparison(self, tmp_path):
        data = [{"method": "Random", "top_delta": 5.0, "mean_delta": 1.0}]
        out_png = tmp_path / "disc.png"
        plot_discovery_comparison(data, output_path=out_png)
        assert out_png.exists()
        assert out_png.stat().st_size > 1000

    def test_plot_budget_tradeoff(self, tmp_path):
        budgets = [0.01, 0.05]
        clean_accs = [93.0, 92.8]
        fault_accs = [84.0, 90.0]
        out_png = tmp_path / "tradeoff.png"
        plot_budget_tradeoff_curve(budgets, clean_accs, fault_accs, output_path=out_png)
        assert out_png.exists()
        assert out_png.stat().st_size > 1000

    def test_plot_radar_evaluation(self, tmp_path):
        cats = ["Clean", "Known", "Unseen"]
        b_scores = [90.0, 70.0, 85.0]
        p_scores = [89.5, 88.0, 89.0]
        out_png = tmp_path / "radar.png"
        plot_radar_evaluation(cats, b_scores, p_scores, output_path=out_png)
        assert out_png.exists()
        assert out_png.stat().st_size > 1000


@pytest.mark.unit
class TestReportCompiler:

    def test_build_research_report(self, tmp_path):
        disc_data = [{"method": "Random", "top_delta": 5.0, "mean_delta": 1.0, "budget": 50, "strategy": "Uniform"}]
        prot_data = [{"model_label": "Baseline", "clean_acc": 93.0, "known_acc": 75.0, "unseen_acc": 88.0, "two_fault_acc": 70.0, "five_fault_acc": 55.0}]
        interaction = {"total_pairs": 10, "num_synergistic": 2, "num_masking": 1, "num_additive": 7, "pct_synergistic": 20.0, "pct_masking": 10.0, "pct_additive": 70.0}

        out_md = tmp_path / "report.md"
        report_text = build_research_report(
            model_name="resnet18",
            baseline_discovery_data=disc_data,
            protection_budget_data=prot_data,
            interaction_summary=interaction,
            output_path=out_md
        )

        assert "VulnShield-DNN" in report_text
        assert "RESNET18" in report_text
        assert out_md.exists()
