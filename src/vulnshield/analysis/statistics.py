"""Scientific Statistical Analysis, Hypothesis Testing, and Ranking Metrics for Publication."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from scipy import stats


@dataclass(frozen=True)
class StatisticalSummary:
    """Summary statistics for empirical experimental distributions."""
    mean: float
    std: float
    median: float
    ci_lower: float
    ci_upper: float
    sample_size: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean": self.mean,
            "std": self.std,
            "median": self.median,
            "ci_95_lower": self.ci_lower,
            "ci_95_upper": self.ci_upper,
            "sample_size": float(self.sample_size)
        }


def compute_distribution_statistics(
    samples: Sequence[float],
    confidence_level: float = 0.95
) -> StatisticalSummary:
    """Compute mean, standard deviation, median, and two-sided bootstrap confidence interval.

    Args:
        samples: Sequence of scalar measurements across independent seeds/trials.
        confidence_level: Desired confidence level (default: 0.95).

    Returns:
        StatisticalSummary object.
    """
    arr = np.asarray(samples, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return StatisticalSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0)
    if n == 1:
        v = float(arr[0])
        return StatisticalSummary(v, 0.0, v, v, v, 1)

    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1))
    median_val = float(np.median(arr))

    # Bootstrap 95% Confidence Interval (10,000 resamples)
    rng = np.random.default_rng(seed=42)
    boot_means = [np.mean(rng.choice(arr, size=n, replace=True)) for _ in range(10000)]
    alpha = (1.0 - confidence_level) / 2.0
    ci_lower = float(np.percentile(boot_means, alpha * 100))
    ci_upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))

    return StatisticalSummary(
        mean=mean_val,
        std=std_val,
        median=median_val,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        sample_size=n
    )


def compute_paired_significance(
    treatment_samples: Sequence[float],
    control_samples: Sequence[float]
) -> Dict[str, Union[float, str, bool]]:
    """Perform paired hypothesis testing (Student's t-test and Wilcoxon signed-rank test).

    Args:
        treatment_samples: Performance measurements from proposed method (e.g. TD3).
        control_samples: Performance measurements from baseline (e.g. Random, Taylor).

    Returns:
        Dict containing t_statistic, p_value_t, wilcoxon_stat, p_value_wilcoxon, cohens_d, is_significant.
    """
    t_arr = np.asarray(treatment_samples, dtype=np.float64)
    c_arr = np.asarray(control_samples, dtype=np.float64)

    if len(t_arr) != len(c_arr):
        raise ValueError(f"Sample length mismatch: treatment has {len(t_arr)}, control has {len(c_arr)}")
    if len(t_arr) < 2:
        return {
            "p_value": 1.0,
            "cohens_d": 0.0,
            "is_significant_p05": False,
            "note": "Insufficient samples for hypothesis testing (N < 2)"
        }

    # 1. Paired t-test
    t_res = stats.ttest_rel(t_arr, c_arr)
    t_stat = float(t_res.statistic)
    p_val_t = float(t_res.pvalue)

    # 2. Wilcoxon signed-rank test (non-parametric fallback)
    diff = t_arr - c_arr
    if np.all(diff == 0):
        p_val_w = 1.0
        w_stat = 0.0
    else:
        try:
            w_res = stats.wilcoxon(diff)
            w_stat = float(w_res.statistic)
            p_val_w = float(w_res.pvalue)
        except Exception:
            w_stat, p_val_w = 0.0, 1.0

    # 3. Effect Size: Cohen's d
    diff_mean = np.mean(diff)
    diff_std = np.std(diff, ddof=1) if np.std(diff, ddof=1) > 1e-9 else 1e-9
    cohens_d = float(diff_mean / diff_std)

    return {
        "t_statistic": t_stat,
        "p_value_parametric": p_val_t,
        "wilcoxon_stat": w_stat,
        "p_value_nonparametric": p_val_w,
        "cohens_d": cohens_d,
        "is_significant_p05": bool(p_val_t < 0.05 or p_val_w < 0.05)
    }


def holm_bonferroni_correction(p_values: Sequence[float], alpha: float = 0.05) -> List[bool]:
    """Apply Holm-Bonferroni step-down correction for multiple hypothesis testing."""
    m = len(p_values)
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    rejected = [False] * m

    for rank, (orig_idx, p_val) in enumerate(indexed_p):
        thresh = alpha / (m - rank)
        if p_val <= thresh:
            rejected[orig_idx] = True
        else:
            break
    return rejected


def compute_discovery_curve_auc(delta_trajectory: Sequence[float]) -> float:
    """Calculate Area Under the Discovery Curve (AUDC) over evaluation steps.

    Higher AUDC indicates faster convergence to critical vulnerabilities under early queries.
    """
    arr = np.asarray(delta_trajectory, dtype=np.float64)
    if len(arr) == 0:
        return 0.0
    # Cumulative maximum over query budget
    cummax = np.maximum.accumulate(arr)
    # Discrete trapezoidal integral
    return float(np.sum(cummax))


def compute_ranking_correlations(
    predicted_ranks: Sequence[float],
    ground_truth_ranks: Sequence[float]
) -> Dict[str, float]:
    """Calculate Spearman's rho and Kendall's tau rank correlation coefficients."""
    p_arr = np.asarray(predicted_ranks, dtype=np.float64)
    g_arr = np.asarray(ground_truth_ranks, dtype=np.float64)

    if len(p_arr) < 3 or len(g_arr) < 3:
        return {"spearman_rho": 0.0, "kendall_tau": 0.0}

    rho, _ = stats.spearmanr(p_arr, g_arr)
    tau, _ = stats.kendalltau(p_arr, g_arr)

    return {
        "spearman_rho": float(rho) if not np.isnan(rho) else 0.0,
        "kendall_tau": float(tau) if not np.isnan(tau) else 0.0
    }
