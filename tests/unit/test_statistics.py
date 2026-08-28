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

    def test_multi_seed_paired_significance_with_holm_bonferroni(self):
        # 5 seeds comparison: TD3 vs 4 baselines
        td3_seeds = [15.2, 16.4, 14.9, 17.1, 15.8]
        random_seeds = [6.1, 7.2, 5.8, 8.0, 6.5]
        activation_seeds = [9.0, 10.1, 8.8, 10.5, 9.4]
        gradient_seeds = [11.2, 12.0, 11.5, 13.0, 12.1]
        ddpg_seeds = [13.1, 14.0, 13.5, 14.8, 13.9]

        p_vals = []
        for bl in [random_seeds, activation_seeds, gradient_seeds, ddpg_seeds]:
            res = compute_paired_significance(td3_seeds, bl)
            assert "p_value_parametric" in res
            assert "cohens_d" in res
            assert res["cohens_d"] > 0  # TD3 is higher in all paired seeds
            p_vals.append(res["p_value_parametric"])

        rejected = holm_bonferroni_correction(p_vals, alpha=0.05)
        assert len(rejected) == 4
        # All comparisons have strong effect sizes and low p-values
        assert all(rejected)
