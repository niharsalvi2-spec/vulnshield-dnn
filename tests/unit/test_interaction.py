"""Unit Tests for Phase 8 — Multi-Channel Fault Interaction Engine."""

from pathlib import Path
import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from vulnshield.interaction.metrics import (
    InteractionType,
    PairwiseInteractionResult,
    compute_interaction_score,
    classify_interaction
)
from vulnshield.interaction.synergy import summarize_interactions
from vulnshield.interaction.evaluator import evaluate_pairwise_interactions
from vulnshield.interaction.visualization import (
    build_interaction_matrix,
    plot_interaction_heatmap
)
from vulnshield.models.resnet import resnet18


@pytest.fixture(scope="module")
def tiny_model():
    m = resnet18(num_classes=10)
    m.eval()
    return m


@pytest.fixture(scope="module")
def eval_loader():
    images = torch.randn(32, 3, 32, 32)
    labels = torch.randint(0, 10, (32,))
    return DataLoader(TensorDataset(images, labels), batch_size=16)


@pytest.mark.unit
class TestInteractionMetrics:

    def test_compute_interaction_score_synergistic(self):
        # E(A)=2%, E(B)=3%, E(A,B)=8% -> I = 8 - (2+3) = +3.0% (Synergy)
        score = compute_interaction_score(delta_a=2.0, delta_b=3.0, delta_joint=8.0)
        assert abs(score - 3.0) < 1e-5
        assert classify_interaction(score) == InteractionType.SYNERGISTIC

    def test_compute_interaction_score_masking(self):
        # E(A)=5%, E(B)=4%, E(A,B)=6% -> I = 6 - (5+4) = -3.0% (Masking)
        score = compute_interaction_score(delta_a=5.0, delta_b=4.0, delta_joint=6.0)
        assert abs(score - (-3.0)) < 1e-5
        assert classify_interaction(score) == InteractionType.MASKING

    def test_compute_interaction_score_additive(self):
        # E(A)=2%, E(B)=3%, E(A,B)=5.2% -> I = 5.2 - 5.0 = +0.2% (Additive)
        score = compute_interaction_score(delta_a=2.0, delta_b=3.0, delta_joint=5.2)
        assert abs(score - 0.2) < 1e-5
        assert classify_interaction(score) == InteractionType.ADDITIVE


@pytest.mark.unit
class TestSynergySummarizer:

    def test_summarize_interactions(self):
        res1 = PairwiseInteractionResult(
            channel_a=("conv1", 0), channel_b=("conv1", 1),
            delta_a=1.0, delta_b=1.0, delta_joint=5.0,
            interaction_score=3.0, interaction_type=InteractionType.SYNERGISTIC
        )
        res2 = PairwiseInteractionResult(
            channel_a=("conv1", 0), channel_b=("conv1", 2),
            delta_a=2.0, delta_b=2.0, delta_joint=2.0,
            interaction_score=-2.0, interaction_type=InteractionType.MASKING
        )
        res3 = PairwiseInteractionResult(
            channel_a=("conv1", 1), channel_b=("conv1", 2),
            delta_a=1.0, delta_b=2.0, delta_joint=3.1,
            interaction_score=0.1, interaction_type=InteractionType.ADDITIVE
        )

        summary = summarize_interactions([res1, res2, res3])
        assert summary.total_pairs == 3
        assert summary.num_synergistic == 1
        assert summary.num_masking == 1
        assert summary.num_additive == 1
        assert summary.max_synergy_pair == res1
        assert summary.max_masking_pair == res2


@pytest.mark.unit
class TestInteractionEvaluator:

    def test_evaluate_pairwise_interactions_execution(self, tiny_model, eval_loader):
        candidates = [("conv1", 0), ("conv1", 1), ("conv1", 2)]
        results = evaluate_pairwise_interactions(
            model=tiny_model,
            channels=candidates,
            dataloader=eval_loader,
            clean_accuracy=10.0,
            device=torch.device("cpu"),
            verbose=False
        )

        # C(3, 2) = 3 pairs
        assert len(results) == 3
        for r in results:
            assert isinstance(r, PairwiseInteractionResult)
            assert r.interaction_score == pytest.approx(
                r.delta_joint - (r.delta_a + r.delta_b), abs=1e-4
            )


@pytest.mark.unit
class TestInteractionVisualization:

    def test_build_interaction_matrix_symmetry(self):
        channels = [("conv1", 0), ("conv1", 1), ("conv1", 2)]
        results = [
            PairwiseInteractionResult(
                channel_a=channels[0], channel_b=channels[1],
                delta_a=1.0, delta_b=1.0, delta_joint=4.0,
                interaction_score=2.0, interaction_type=InteractionType.SYNERGISTIC
            ),
            PairwiseInteractionResult(
                channel_a=channels[0], channel_b=channels[2],
                delta_a=1.0, delta_b=2.0, delta_joint=1.5,
                interaction_score=-1.5, interaction_type=InteractionType.MASKING
            ),
            PairwiseInteractionResult(
                channel_a=channels[1], channel_b=channels[2],
                delta_a=1.0, delta_b=2.0, delta_joint=3.0,
                interaction_score=0.0, interaction_type=InteractionType.ADDITIVE
            )
        ]

        matrix, labels = build_interaction_matrix(results, channels)
        assert matrix.shape == (3, 3)
        assert matrix[0, 1] == matrix[1, 0] == 2.0
        assert matrix[0, 2] == matrix[2, 0] == -1.5
        assert matrix[1, 2] == matrix[2, 1] == 0.0
        assert matrix[0, 0] == 0.0  # diagonal

    def test_plot_interaction_heatmap_generates_file(self, tmp_path):
        matrix = np.array([[0.0, 2.5], [2.5, 0.0]])
        labels = ["ch0", "ch1"]
        out_png = tmp_path / "heatmap_test.png"

        plot_interaction_heatmap(matrix, labels, output_path=out_png)
        assert out_png.exists()
        assert out_png.stat().st_size > 1000
