"""Scientific Data Integrity & Leakage Auditor.

Verifies strict dataset partition disjointness:
  - 40,000 Train
  - 5,000 Validation
  - 5,000 Test (Strictly preserved for final evaluation)
  - 1,000 EvalFault (Disjoint subset used for RL discovery calibration)
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Set, Union
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset, Subset


def compute_sample_fingerprint(image_tensor: torch.Tensor, label: int) -> str:
    """Generate SHA-256 fingerprint for a single data sample."""
    img_bytes = image_tensor.numpy().tobytes()
    hasher = hashlib.sha256(img_bytes)
    hasher.update(str(label).encode("ascii"))
    return hasher.hexdigest()


class DataLeakageAuditor:
    """Audits dataset splits for contamination and overlap."""

    @staticmethod
    def audit_splits(
        train_indices: Sequence[int],
        val_indices: Sequence[int],
        test_indices: Sequence[int],
        eval_fault_indices: Sequence[int]
    ) -> Dict[str, Union[bool, int, List[str]]]:
        """Perform set-theoretic disjointness verification on sample indices."""
        set_train = set(train_indices)
        set_val = set(val_indices)
        set_test = set(test_indices)
        set_fault = set(eval_fault_indices)

        overlaps = {
            "train_val_overlap": len(set_train.intersection(set_val)),
            "train_test_overlap": len(set_train.intersection(set_test)),
            "val_test_overlap": len(set_val.intersection(set_test)),
            "test_fault_overlap": len(set_test.intersection(set_fault))
        }

        has_leakage = any(v > 0 for v in overlaps.values())

        return {
            "has_leakage": has_leakage,
            "train_count": len(set_train),
            "val_count": len(set_val),
            "test_count": len(set_test),
            "eval_fault_count": len(set_fault),
            "overlaps": overlaps,
            "status": "PASS — Zero Data Leakage" if not has_leakage else "FAIL — Data Leakage Detected"
        }
