"""Multi-Fault Synergy Analysis Utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from vulnshield.interaction.metrics import (
    PairwiseInteractionResult,
    InteractionType,
    compute_interaction_score,
    classify_interaction
)


@dataclass
class InteractionSummary:
    """Summary of pairwise interaction analysis across a channel pool."""
    total_pairs: int
    num_synergistic: int
    num_masking: int
    num_additive: int
    max_synergy_pair: Optional[PairwiseInteractionResult] = None
    max_masking_pair: Optional[PairwiseInteractionResult] = None
    results: List[PairwiseInteractionResult] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "total_pairs": self.total_pairs,
            "num_synergistic": self.num_synergistic,
            "num_masking": self.num_masking,
            "num_additive": self.num_additive,
            "pct_synergistic": round(self.num_synergistic / max(self.total_pairs, 1) * 100, 2),
            "pct_masking": round(self.num_masking / max(self.total_pairs, 1) * 100, 2),
            "pct_additive": round(self.num_additive / max(self.total_pairs, 1) * 100, 2),
            "max_synergy": self.max_synergy_pair.to_dict() if self.max_synergy_pair else None,
            "max_masking": self.max_masking_pair.to_dict() if self.max_masking_pair else None
        }


def summarize_interactions(
    results: Sequence[PairwiseInteractionResult]
) -> InteractionSummary:
    """Aggregate individual pairwise interaction results into summary statistics."""
    total = len(results)
    if total == 0:
        return InteractionSummary(total_pairs=0, num_synergistic=0, num_masking=0, num_additive=0)

    num_syn = sum(1 for r in results if r.interaction_type == InteractionType.SYNERGISTIC)
    num_mask = sum(1 for r in results if r.interaction_type == InteractionType.MASKING)
    num_add = sum(1 for r in results if r.interaction_type == InteractionType.ADDITIVE)

    sorted_by_syn = sorted(results, key=lambda r: r.interaction_score, reverse=True)
    max_syn = sorted_by_syn[0] if sorted_by_syn and sorted_by_syn[0].interaction_score > 0 else None
    max_mask = sorted_by_syn[-1] if sorted_by_syn and sorted_by_syn[-1].interaction_score < 0 else None

    return InteractionSummary(
        total_pairs=total,
        num_synergistic=num_syn,
        num_masking=num_mask,
        num_additive=num_add,
        max_synergy_pair=max_syn,
        max_masking_pair=max_mask,
        results=list(results)
    )
