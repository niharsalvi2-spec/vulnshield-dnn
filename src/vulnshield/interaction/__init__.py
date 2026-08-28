"""VulnShield-DNN Multi-Fault Interaction Package."""

from vulnshield.interaction.metrics import (
    InteractionType,
    PairwiseInteractionResult,
    compute_interaction_score,
    classify_interaction
)
from vulnshield.interaction.synergy import (
    InteractionSummary,
    summarize_interactions
)
from vulnshield.interaction.evaluator import evaluate_pairwise_interactions
from vulnshield.interaction.visualization import (
    build_interaction_matrix,
    plot_interaction_heatmap
)

__all__ = [
    "InteractionType",
    "PairwiseInteractionResult",
    "compute_interaction_score",
    "classify_interaction",
    "InteractionSummary",
    "summarize_interactions",
    "evaluate_pairwise_interactions",
    "build_interaction_matrix",
    "plot_interaction_heatmap"
]
