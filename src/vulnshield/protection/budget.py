"""Protection Budget Allocation and Channel Selection Utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple, Union

from vulnshield.fault_injection.fault_injector import FaultSpec


@dataclass
class ProtectionBudget:
    """Represents an allocated protection budget and its selected channel subset."""
    percentage: float                  # e.g., 0.01, 0.03, 0.05, 0.10
    num_channels: int                  # Absolute channel count
    selected_channels: List[FaultSpec] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Union[float, int, list]]:
        return {
            "budget_percentage": self.percentage,
            "budget_pct_label": f"{int(self.percentage * 100)}%",
            "num_channels": self.num_channels,
            "selected_channels": [
                f"{layer}_c{c}" for layer, c in self.selected_channels
            ]
        }


def calculate_budget_channel_count(total_channels: int, budget_pct: float) -> int:
    """Compute the number of channels corresponding to a given budget percentage.

    Args:
        total_channels: Total convolutional channels in the network.
        budget_pct: Budget percentage (e.g. 0.01 for 1%).

    Returns:
        Number of channels (at least 1).
    """
    if budget_pct <= 0.0 or budget_pct > 1.0:
        raise ValueError(f"budget_pct must be in (0.0, 1.0], got {budget_pct}")
    return max(1, int(round(total_channels * budget_pct)))


def select_top_k_channels(
    ranked_discoveries: Sequence[Union[Dict, Tuple[str, int]]],
    num_channels: int
) -> List[FaultSpec]:
    """Extract top-K channels from ranked discovery list.

    Args:
        ranked_discoveries: List of discovered channels sorted by vulnerability.
        num_channels: Number of top channels to select.

    Returns:
        List of (layer_name, channel_idx) tuples.
    """
    selected: List[FaultSpec] = []
    for item in ranked_discoveries:
        if isinstance(item, dict):
            ch = (item["layer_name"], item["channel_idx"])
        elif isinstance(item, tuple):
            ch = (item[0], item[1])
        else:
            continue

        if ch not in selected:
            selected.append(ch)
        if len(selected) >= num_channels:
            break

    return selected
