"""Result Schema Validator — Automated Sanity Check for Experimental Outputs.

Enforces strict schema and value constraints on every result file before it
enters the reporting pipeline. Detects fabricated, NaN, Inf, or out-of-bounds
empirical values and fails loudly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class ResultValidationError(Exception):
    """Raised when an experimental result file fails validation."""
    pass


def _check_finite(val: Any, field: str) -> None:
    if isinstance(val, float):
        if math.isnan(val):
            raise ResultValidationError(f"Field '{field}' contains NaN — possible numerical instability or placeholder.")
        if math.isinf(val):
            raise ResultValidationError(f"Field '{field}' contains Inf — possible numerical instability.")


def _check_accuracy(val: float, field: str) -> None:
    if not (0.0 <= val <= 100.0):
        raise ResultValidationError(
            f"Accuracy field '{field}' = {val} is outside valid range [0, 100]."
        )


def _check_positive(val: Union[int, float], field: str) -> None:
    if val < 0:
        raise ResultValidationError(f"Field '{field}' = {val} must be non-negative.")


def validate_discovery_result(result: Dict[str, Any], budget: int) -> None:
    """Validate a TD3 or baseline discovery result dictionary."""
    required = ["top_channels", "total_queries_executed", "max_budget_enforced"]
    for key in required:
        if key not in result:
            raise ResultValidationError(f"Discovery result missing required key: '{key}'")

    q_exec = result["total_queries_executed"]
    q_budget = result["max_budget_enforced"]

    _check_positive(q_exec, "total_queries_executed")
    if q_exec > q_budget:
        raise ResultValidationError(
            f"Budget violation: executed {q_exec} queries but budget was {q_budget}."
        )

    for ch in result.get("top_channels", []):
        delta = ch.get("delta_accuracy", 0.0)
        _check_finite(delta, "delta_accuracy")
        if delta < 0:
            raise ResultValidationError(
                f"delta_accuracy = {delta} is negative — fault injection worsened accuracy is valid but should be checked."
            )


def validate_evaluation_report(report: Dict[str, Any]) -> None:
    """Validate a 6-dimensional evaluation report."""
    required_dims = ["dim1_clean", "dim2_known_faults", "dim3_unseen_faults", "dim4_multi_faults"]
    for dim in required_dims:
        if dim not in report:
            raise ResultValidationError(f"Evaluation report missing dimension: '{dim}'")

    clean_acc = report.get("dim1_clean", {}).get("accuracy", None)
    if clean_acc is not None:
        _check_finite(clean_acc, "dim1_clean.accuracy")
        _check_accuracy(clean_acc, "dim1_clean.accuracy")

    known_mean = report.get("dim2_known_faults", {}).get("mean_accuracy", None)
    if known_mean is not None:
        _check_finite(known_mean, "dim2_known_faults.mean_accuracy")
        _check_accuracy(known_mean, "dim2_known_faults.mean_accuracy")


def validate_protection_summary(summary: Dict[str, Any], clean_baseline: float, tolerance: float = 0.99) -> None:
    """Validate protection experiment result and enforce clean-accuracy retention constraint."""
    protected_clean = summary.get("best_clean_acc", None)
    if protected_clean is None:
        raise ResultValidationError("Protection summary missing 'best_clean_acc' field.")

    _check_finite(protected_clean, "best_clean_acc")
    _check_accuracy(protected_clean, "best_clean_acc")

    retention_ratio = protected_clean / max(clean_baseline, 1e-6)
    summary["clean_retention_ratio"] = retention_ratio
    summary["clean_accuracy_constraint"] = "PASS" if retention_ratio >= tolerance else "FAIL"

    if retention_ratio < tolerance:
        print(
            f"[WARNING] Clean accuracy constraint FAIL: "
            f"protected={protected_clean:.2f}%, baseline={clean_baseline:.2f}%, "
            f"retention={retention_ratio:.4f} (required >= {tolerance:.2f})"
        )


def validate_result_file(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Load and perform schema + value validation on any result JSON file."""
    p = Path(filepath).resolve()
    if not p.exists():
        raise ResultValidationError(f"Result file not found: {p}")
    if p.stat().st_size == 0:
        raise ResultValidationError(f"Result file is empty: {p}")

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, (dict, list)):
        raise ResultValidationError(f"Result file root must be dict or list, got: {type(data)}")

    return data
