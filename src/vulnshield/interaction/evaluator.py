"""Multi-Channel Combinatorial Fault Evaluator."""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from vulnshield.fault_injection.fault_injector import FaultInjector, FaultSpec
from vulnshield.training.evaluator import evaluate_model
from vulnshield.interaction.metrics import (
    PairwiseInteractionResult,
    InteractionType,
    compute_interaction_score,
    classify_interaction
)


def evaluate_pairwise_interactions(
    model: nn.Module,
    channels: Sequence[FaultSpec],
    dataloader: DataLoader,
    clean_accuracy: float,
    device: Optional[torch.device] = None,
    synergy_threshold: float = 1.0,
    masking_threshold: float = -1.0,
    verbose: bool = True
) -> List[PairwiseInteractionResult]:
    """Evaluate all pairwise simultaneous fault combinations among candidate channels.

    Args:
        model: Pre-trained target neural network.
        channels: List of (layer_name, channel_idx) candidate fault targets.
        dataloader: Evaluation DataLoader.
        clean_accuracy: Pre-computed baseline clean model accuracy (%).
        device: Compute device.
        synergy_threshold: Minimum positive interaction score for synergistic failure.
        masking_threshold: Maximum negative interaction score for masking effect.
        verbose: Display progress bar.

    Returns:
        List of PairwiseInteractionResult objects for all C(N, 2) pairs.
    """
    dev = device or torch.device("cpu")
    injector = FaultInjector(model)

    # 1. Step 1: Pre-compute individual fault degradations E(A)
    individual_deltas: Dict[FaultSpec, float] = {}
    if verbose:
        print(f"[*] Step 1/2: Pre-evaluating {len(channels)} individual channel fault impacts...")

    for ch in channels:
        with injector.inject([ch]):
            res = evaluate_model(model, dataloader, device=dev)
        individual_deltas[ch] = clean_accuracy - res.accuracy

    # 2. Step 2: Evaluate all pairwise combinations C(N, 2)
    pairs = list(itertools.combinations(channels, 2))
    results: List[PairwiseInteractionResult] = []

    pbar = tqdm(pairs, desc="Pairwise Multi-Fault Evaluation", leave=False, disable=not verbose)
    for ch_a, ch_b in pbar:
        with injector.inject([ch_a, ch_b]):
            res = evaluate_model(model, dataloader, device=dev)
        delta_joint = clean_accuracy - res.accuracy

        delta_a = individual_deltas[ch_a]
        delta_b = individual_deltas[ch_b]
        score = compute_interaction_score(delta_a, delta_b, delta_joint)
        itype = classify_interaction(score, synergy_threshold, masking_threshold)

        results.append(PairwiseInteractionResult(
            channel_a=ch_a,
            channel_b=ch_b,
            delta_a=delta_a,
            delta_b=delta_b,
            delta_joint=delta_joint,
            interaction_score=score,
            interaction_type=itype
        ))

    results.sort(key=lambda r: r.interaction_score, reverse=True)
    return results
