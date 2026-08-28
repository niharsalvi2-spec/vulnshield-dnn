"""Visualization Utilities for Fault Interaction Matrices and Synergy Heatmaps."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple, Union
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from vulnshield.fault_injection.fault_injector import FaultSpec
from vulnshield.interaction.metrics import PairwiseInteractionResult


def build_interaction_matrix(
    results: Sequence[PairwiseInteractionResult],
    channels: Sequence[FaultSpec]
) -> Tuple[np.ndarray, List[str]]:
    """Construct an N x N symmetric interaction score matrix from pairwise results.

    Args:
        results: Sequence of PairwiseInteractionResult objects.
        channels: Ordered sequence of candidate fault targets.

    Returns:
        Tuple of (interaction_matrix, channel_labels).
    """
    n = len(channels)
    ch_to_idx = {ch: idx for idx, ch in enumerate(channels)}
    matrix = np.zeros((n, n), dtype=np.float32)

    for r in results:
        if r.channel_a in ch_to_idx and r.channel_b in ch_to_idx:
            i = ch_to_idx[r.channel_a]
            j = ch_to_idx[r.channel_b]
            matrix[i, j] = r.interaction_score
            matrix[j, i] = r.interaction_score  # Symmetric

    channel_labels = [f"{layer.split('.')[-1]}_c{c}" for layer, c in channels]
    return matrix, channel_labels


def plot_interaction_heatmap(
    matrix: np.ndarray,
    channel_labels: List[str],
    output_path: Union[str, Path],
    title: str = "Pairwise Fault Interaction Matrix I(A, B)",
    cmap: str = "coolwarm"
) -> None:
    """Render and save a publication-quality interaction heatmap.

    Args:
        matrix: N x N interaction matrix.
        channel_labels: Labels for rows and columns.
        output_path: Destination image path (.png).
        title: Figure title.
        cmap: Matplotlib colormap name.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")

    # Symmetric limit for coolwarm centering around 0
    vmax = max(abs(np.nanmax(matrix)), abs(np.nanmin(matrix)), 1.0)
    vmin = -vmax

    ax = sns.heatmap(
        matrix,
        xticklabels=channel_labels,
        yticklabels=channel_labels,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        center=0.0,
        annot=len(channel_labels) <= 15,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={"label": "Interaction Score I(A, B) (%)"}
    )

    plt.title(title, fontsize=14, pad=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out_file), dpi=300)
    plt.close()
