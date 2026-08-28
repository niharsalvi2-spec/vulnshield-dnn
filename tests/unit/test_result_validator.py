"""Unit Tests for Result Schema Validation and Sanity Checks."""

import pytest
from vulnshield.validation.result_validator import (
    validate_discovery_result,
    validate_evaluation_report,
    validate_protection_summary,
    ResultValidationError
)


@pytest.mark.unit
class TestResultValidator:

    def test_valid_discovery_result(self):
        result = {
            "top_channels": [{"layer_name": "layer1.conv1", "channel_idx": 5, "delta_accuracy": 8.4}],
            "total_queries_executed": 50,
            "max_budget_enforced": 50
        }
        validate_discovery_result(result, budget=50)  # Should not raise

    def test_discovery_budget_violation(self):
        result = {
            "top_channels": [],
            "total_queries_executed": 75,
            "max_budget_enforced": 50
        }
        with pytest.raises(ResultValidationError, match="Budget violation"):
            validate_discovery_result(result, budget=50)

    def test_discovery_nan_delta(self):
        result = {
            "top_channels": [{"delta_accuracy": float("nan")}],
            "total_queries_executed": 10,
            "max_budget_enforced": 50
        }
        with pytest.raises(ResultValidationError, match="NaN"):
            validate_discovery_result(result, budget=50)

    def test_valid_evaluation_report(self):
        report = {
            "dim1_clean": {"accuracy": 92.8},
            "dim2_known_faults": {"mean_accuracy": 85.3},
            "dim3_unseen_faults": {"mean_accuracy": 88.1},
            "dim4_multi_faults": {"2_faults": 79.2, "3_faults": 70.1}
        }
        validate_evaluation_report(report)  # Should not raise

    def test_invalid_accuracy_out_of_range(self):
        report = {
            "dim1_clean": {"accuracy": 150.0},  # > 100, invalid
            "dim2_known_faults": {"mean_accuracy": 80.0},
            "dim3_unseen_faults": {"mean_accuracy": 78.0},
            "dim4_multi_faults": {}
        }
        with pytest.raises(ResultValidationError, match="outside valid range"):
            validate_evaluation_report(report)

    def test_clean_accuracy_constraint_pass(self):
        summary = {"best_clean_acc": 92.5}
        validate_protection_summary(summary, clean_baseline=93.0, tolerance=0.99)
        assert summary["clean_accuracy_constraint"] == "PASS"

    def test_clean_accuracy_constraint_fail(self):
        summary = {"best_clean_acc": 70.0}
        validate_protection_summary(summary, clean_baseline=93.0, tolerance=0.99)
        assert summary["clean_accuracy_constraint"] == "FAIL"
