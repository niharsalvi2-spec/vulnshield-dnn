"""VulnShield-DNN Protection Package — Fault-Aware Fine-Tuning."""

from vulnshield.protection.budget import (
    ProtectionBudget,
    calculate_budget_channel_count,
    select_top_k_channels
)
from vulnshield.protection.losses import FaultAwareLoss
from vulnshield.protection.regularizer import WeightDriftRegularizer
from vulnshield.protection.fine_tuning import (
    FaultAwareTrainer,
    ProtectionTrainingConfig
)

__all__ = [
    "ProtectionBudget",
    "calculate_budget_channel_count",
    "select_top_k_channels",
    "FaultAwareLoss",
    "WeightDriftRegularizer",
    "FaultAwareTrainer",
    "ProtectionTrainingConfig"
]
