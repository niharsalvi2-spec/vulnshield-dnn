"""VulnShield-DNN Comprehensive Evaluation Suite Package."""

from vulnshield.evaluation.metrics import ComprehensiveEvaluationReport
from vulnshield.evaluation.clean_accuracy import evaluate_clean_preservation
from vulnshield.evaluation.fault_evaluator import (
    evaluate_channel_fault_set,
    evaluate_unseen_channel_generalization,
    evaluate_simultaneous_multi_faults
)
from vulnshield.evaluation.bit_flip import (
    flip_float32_bit,
    evaluate_bit_flip_robustness,
    BIT_POSITIONS
)
from vulnshield.evaluation.adversarial import (
    fgsm_attack,
    pgd_attack,
    evaluate_adversarial_robustness
)

__all__ = [
    "ComprehensiveEvaluationReport",
    "evaluate_clean_preservation",
    "evaluate_channel_fault_set",
    "evaluate_unseen_channel_generalization",
    "evaluate_simultaneous_multi_faults",
    "flip_float32_bit",
    "evaluate_bit_flip_robustness",
    "BIT_POSITIONS",
    "fgsm_attack",
    "pgd_attack",
    "evaluate_adversarial_robustness"
]
