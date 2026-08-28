"""Comprehensive Multi-Dimensional Evaluation Metrics and Report Container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class ComprehensiveEvaluationReport:
    """Consolidated results across all 6 evaluation dimensions for a model."""
    model_name: str
    checkpoint_name: str
    
    # Dimension 1: Clean Accuracy
    clean_accuracy: float
    clean_loss: float
    
    # Dimension 2: Known Protected Channel Fault Accuracy
    known_fault_accuracy: float
    known_fault_drop: float
    
    # Dimension 3: Unseen Channel Fault Generalization
    unseen_fault_accuracy: float
    unseen_fault_drop: float
    
    # Dimension 4: Simultaneous Multi-Fault Stress Test
    multi_fault_accuracies: Dict[int, float] = field(default_factory=dict)
    
    # Dimension 5: Physical Bit-Flip Robustness
    bit_flip_accuracies: Dict[str, float] = field(default_factory=dict)
    
    # Dimension 6: Decoupled Adversarial Comparison
    fgsm_accuracy: Optional[float] = None
    pgd_accuracy: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "checkpoint_name": self.checkpoint_name,
            "dim1_clean": {
                "accuracy": round(self.clean_accuracy, 2),
                "loss": round(self.clean_loss, 4)
            },
            "dim2_known_faults": {
                "mean_accuracy": round(self.known_fault_accuracy, 2),
                "mean_drop": round(self.known_fault_drop, 2)
            },
            "dim3_unseen_faults": {
                "mean_accuracy": round(self.unseen_fault_accuracy, 2),
                "mean_drop": round(self.unseen_fault_drop, 2)
            },
            "dim4_multi_faults": {
                f"{k}_faults": round(v, 2) for k, v in self.multi_fault_accuracies.items()
            },
            "dim5_bit_flip": {
                k: round(v, 2) for k, v in self.bit_flip_accuracies.items()
            },
            "dim6_adversarial": {
                "fgsm": round(self.fgsm_accuracy, 2) if self.fgsm_accuracy is not None else None,
                "pgd20": round(self.pgd_accuracy, 2) if self.pgd_accuracy is not None else None
            }
        }
