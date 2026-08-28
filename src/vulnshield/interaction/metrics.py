"""Interaction Metrics and Synergy Classification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple, Union


class InteractionType(str, Enum):
    """Categorical classification of multi-fault interaction."""
    SYNERGISTIC = "synergistic"  # Compounding failure: I(A, B) > threshold
    MASKING = "masking"          # Antagonistic masking: I(A, B) < -threshold
    ADDITIVE = "additive"        # Independent failure: |I(A, B)| <= threshold


@dataclass
class PairwiseInteractionResult:
    """Stores full evaluation details for a pairwise fault interaction."""
    channel_a: Tuple[str, int]
    channel_b: Tuple[str, int]
    delta_a: float
    delta_b: float
    delta_joint: float
    interaction_score: float
    interaction_type: InteractionType

    def to_dict(self) -> Dict[str, Union[str, float, list]]:
        return {
            "channel_a": f"{self.channel_a[0]}_c{self.channel_a[1]}",
            "channel_b": f"{self.channel_b[0]}_c{self.channel_b[1]}",
            "delta_a": round(self.delta_a, 4),
            "delta_b": round(self.delta_b, 4),
            "delta_joint": round(self.delta_joint, 4),
            "interaction_score": round(self.interaction_score, 4),
            "interaction_type": self.interaction_type.value
        }


def compute_interaction_score(delta_a: float, delta_b: float, delta_joint: float) -> float:
    """Calculate the pairwise interaction score I(A, B) = E(A, B) - [E(A) + E(B)].

    Args:
        delta_a: Accuracy degradation from channel A alone (E(A)).
        delta_b: Accuracy degradation from channel B alone (E(B)).
        delta_joint: Accuracy degradation from simultaneous faults (E(A, B)).

    Returns:
        Interaction score float.
    """
    return delta_joint - (delta_a + delta_b)


def classify_interaction(
    interaction_score: float,
    synergy_threshold: float = 1.0,
    masking_threshold: float = -1.0
) -> InteractionType:
    """Classify interaction into Synergistic, Masking, or Additive."""
    if interaction_score > synergy_threshold:
        return InteractionType.SYNERGISTIC
    elif interaction_score < masking_threshold:
        return InteractionType.MASKING
    else:
        return InteractionType.ADDITIVE
