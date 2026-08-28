"""Unit Tests for Phase 9 — Protection Engine (Fault-Aware Fine-Tuning)."""

from pathlib import Path
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
class TestProtectionBudgets:

    def test_budget_channel_counts_resnet18(self):
        total_channels = 4800
        assert calculate_budget_channel_count(total_channels, 0.01) == 48
        assert calculate_budget_channel_count(total_channels, 0.03) == 144
        assert calculate_budget_channel_count(total_channels, 0.05) == 240
        assert calculate_budget_channel_count(total_channels, 0.10) == 480

    def test_budget_channel_counts_vgg16(self):
        total_channels = 4224
        assert calculate_budget_channel_count(total_channels, 0.01) == 42
        assert calculate_budget_channel_count(total_channels, 0.03) == 127
        assert calculate_budget_channel_count(total_channels, 0.05) == 211
        assert calculate_budget_channel_count(total_channels, 0.10) == 422

    def test_invalid_budget_pct_raises(self):
        with pytest.raises(ValueError):
            calculate_budget_channel_count(4800, -0.05)
        with pytest.raises(ValueError):
            calculate_budget_channel_count(4800, 1.5)

    def test_select_top_k_channels(self):
        discoveries = [
            {"layer_name": "conv1", "channel_idx": 5, "delta_accuracy": 15.2},
            {"layer_name": "conv1", "channel_idx": 12, "delta_accuracy": 12.1},
            {"layer_name": "layer1.0.conv1", "channel_idx": 3, "delta_accuracy": 9.4},
            {"layer_name": "layer2.0.conv1", "channel_idx": 10, "delta_accuracy": 8.0}
        ]
        top2 = select_top_k_channels(discoveries, num_channels=2)
        assert len(top2) == 2
        assert top2[0] == ("conv1", 5)
        assert top2[1] == ("conv1", 12)


@pytest.mark.unit
class TestFaultAwareLoss:

    def test_loss_computation(self):
        crit = FaultAwareLoss(alpha=0.5, beta=0.5)
        clean_logits = torch.randn(8, 10)
        fault_logits = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))

        tot_loss, l_clean, l_fault = crit(clean_logits, fault_logits, targets)
        assert tot_loss.ndim == 0
        assert l_clean.ndim == 0
        assert l_fault.ndim == 0
        assert abs(tot_loss.item() - 0.5 * (l_clean.item() + l_fault.item())) < 1e-4

    def test_gradients_flow(self):
        crit = FaultAwareLoss(alpha=0.4, beta=0.6)
        clean_logits = torch.randn(4, 10, requires_grad=True)
        fault_logits = torch.randn(4, 10, requires_grad=True)
        targets = torch.tensor([0, 1, 2, 3])

        tot_loss, _, _ = crit(clean_logits, fault_logits, targets)
        tot_loss.backward()

        assert clean_logits.grad is not None
        assert fault_logits.grad is not None


@pytest.mark.unit
class TestWeightDriftRegularizer:

    def test_zero_drift_on_identical_model(self, tiny_model):
        reg = WeightDriftRegularizer(tiny_model, lambda_drift=1e-3)
        penalty = reg.compute_penalty(tiny_model)
        assert penalty.item() == 0.0

    def test_positive_drift_on_modified_weights(self):
        m1 = nn.Sequential(nn.Linear(4, 4))
        reg = WeightDriftRegularizer(m1, lambda_drift=1.0)

        m2 = nn.Sequential(nn.Linear(4, 4))
        # perturb weights
        with torch.no_grad():
            m2[0].weight.add_(1.0)

        penalty = reg.compute_penalty(m2)
        assert penalty.item() > 0.0


@pytest.mark.unit
class TestFaultAwareTrainer:

    def test_trainer_2_epoch_execution(self, eval_loader, tmp_path):
        m = resnet18(num_classes=10)
        protected_channels = [("conv1", 0), ("conv1", 1)]

        cfg = ProtectionTrainingConfig(
            epochs=2,
            learning_rate=0.01,
            optimizer_name="sgd",
            scheduler_name="cosine"
        )
        trainer = FaultAwareTrainer(
            model=m,
            protected_channels=protected_channels,
            config=cfg,
            device=torch.device("cpu")
        )

        results = trainer.fit(
            train_loader=eval_loader,
            val_loader=eval_loader,
            eval_fault_loader=eval_loader,
            checkpoint_dir=tmp_path / "protected_ckpts",
            checkpoint_name="test_prot"
        )

        assert "best_combined_score" in results
        assert len(trainer.history["clean_loss"]) == 2
        assert len(trainer.history["fault_loss"]) == 2
        assert len(trainer.history["val_fault_acc"]) == 2
