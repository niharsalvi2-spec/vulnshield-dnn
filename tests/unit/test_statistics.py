"""Unit Tests for Statistical Analysis and Hypothesis Testing Module."""

import pytest
import numpy as np

from vulnshield.analysis.statistics import (
    compute_distribution_statistics,
    compute_paired_significance,
    holm_bonferroni_correction,
    compute_discovery_curve_auc,
    compute_ranking_correlations
)


@pytest.mark.unit
class TestStatisticalAnalysis:

    def test_compute_distribution_statistics(self):
        samples = [10.0, 12.0, 11.0, 13.0, 10.5]
        stats_res = compute_distribution_statistics(samples, confidence_level=0.95)

        assert abs(stats_res.mean - 11.3) < 1e-4
        assert stats_res.sample_size == 5
        assert stats_res.ci_lower <= stats_res.mean <= stats_res.ci_upper

    def test_paired_significance_testing(self):
        # Treatment method outperforms control across 5 random seeds
        treatment = [15.2, 16.1, 14.8, 17.0, 15.5]
        control = [8.1, 9.0, 7.5, 9.2, 8.4]

        res = compute_paired_significance(treatment, control)
        assert res["is_significant_p05"] is True
        assert res["cohens_d"] > 2.0  # Very large effect size
        assert res["p_value_parametric"] < 0.01

    def test_holm_bonferroni_correction(self):
        # 3 hypotheses: 2 significant, 1 non-significant
        p_vals = [0.001, 0.01, 0.20]
        rejected = holm_bonferroni_correction(p_vals, alpha=0.05)
        assert rejected == [True, True, False]

    def test_discovery_curve_auc(self):
        # Monotonically increasing rewards
        traj = [2.0, 5.0, 5.0, 10.0]
        auc = compute_discovery_curve_auc(traj)
        assert auc == (2.0 + 5.0 + 5.0 + 10.0)

    def test_ranking_correlations(self):
        pred = [1.0, 2.0, 3.0, 4.0, 5.0]
        gt = [1.0, 2.0, 3.0, 4.0, 5.0]
        corr = compute_ranking_correlations(pred, gt)
        assert abs(corr["spearman_rho"] - 1.0) < 1e-5
        assert abs(corr["kendall_tau"] - 1.0) < 1e-5
