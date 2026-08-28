"""Unit Tests for Phase 10 — Comprehensive Evaluation Suite."""

from pathlib import Path
import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
class TestCleanPreservation:

    def test_clean_preservation_drop_calculation(self, tiny_model, eval_loader):
        res, drop, passed = evaluate_clean_preservation(
            tiny_model, eval_loader, baseline_clean_accuracy=10.0, device=torch.device("cpu"), max_tolerable_drop=1.0
        )
        assert isinstance(drop, float)
        assert isinstance(passed, bool)
        assert abs(drop - (10.0 - res.accuracy)) < 1e-4


@pytest.mark.unit
class TestFaultEvaluators:

    def test_evaluate_channel_fault_set(self, tiny_model, eval_loader):
        channels = [("conv1", 0), ("conv1", 1)]
        mean_acc, mean_drop, detailed = evaluate_channel_fault_set(
            tiny_model, channels, eval_loader, clean_accuracy=10.0, device=torch.device("cpu")
        )
        assert len(detailed) == 2
        assert 0.0 <= mean_acc <= 100.0

    def test_evaluate_unseen_generalization(self, tiny_model, eval_loader):
        from vulnshield.fault_injection.fault_injector import FaultInjector
        injectable = FaultInjector(tiny_model).list_injectable_layers()
        protected = [("conv1", 0), ("conv1", 1)]

        mean_acc, mean_drop, detailed = evaluate_unseen_channel_generalization(
            tiny_model, protected, injectable, eval_loader, clean_accuracy=10.0, num_unseen_samples=4, device=torch.device("cpu")
        )
        assert len(detailed) == 4
        # None of the evaluated channels should be in the protected set
        for item in detailed:
            assert (item["layer_name"], item["channel_idx"]) not in protected

    def test_simultaneous_multi_faults(self, tiny_model, eval_loader):
        from vulnshield.fault_injection.fault_injector import FaultInjector
        injectable = FaultInjector(tiny_model).list_injectable_layers()
        res = evaluate_simultaneous_multi_faults(
            tiny_model, injectable, eval_loader, fault_counts=[2], trials_per_count=2, device=torch.device("cpu")
        )
        assert 2 in res
        assert 0.0 <= res[2] <= 100.0


@pytest.mark.unit
class TestBitFlipSimulation:

    def test_flip_sign_bit_inverts_sign(self):
        val = torch.tensor(3.5, dtype=torch.float32)
        flipped = flip_float32_bit(val, BIT_POSITIONS["sign"])
        assert flipped.item() == -3.5

    def test_flip_exponent_bit_changes_magnitude(self):
        val = torch.tensor(1.0, dtype=torch.float32)
        flipped = flip_float32_bit(val, BIT_POSITIONS["exponent"])
        # Inverting bit 27 flips the exponent field -> causes sharp change
        assert abs(flipped.item() - 1.0) > 0.5

    def test_evaluate_bit_flip_robustness(self, tiny_model, eval_loader):
        res = evaluate_bit_flip_robustness(
            tiny_model, eval_loader, target_bits=["sign"], flips_per_layer=2, device=torch.device("cpu")
        )
        assert "sign" in res
        assert 0.0 <= res["sign"] <= 100.0


@pytest.mark.unit
class TestAdversarialAttacks:

    def test_fgsm_perturbation_bound(self, tiny_model):
        images = torch.randn(4, 3, 32, 32)
        labels = torch.randint(0, 10, (4,))
        eps = 8.0 / 255.0

        adv = fgsm_attack(tiny_model, images, labels, epsilon=eps)
        assert adv.shape == images.shape
        diff = (adv - images).abs()
        # L_inf bound on perturbation
        assert diff.max().item() <= eps + 1e-4

    def test_pgd_perturbation_bound(self, tiny_model):
        images = torch.randn(4, 3, 32, 32)
        labels = torch.randint(0, 10, (4,))
        eps = 8.0 / 255.0

        adv = pgd_attack(tiny_model, images, labels, epsilon=eps, alpha=2.0/255.0, steps=3)
        assert adv.shape == images.shape
        diff = (adv - images).abs()
        assert diff.max().item() <= eps + 1e-4


@pytest.mark.unit
class TestEvaluationReportStructure:

    def test_report_to_dict(self):
        rep = ComprehensiveEvaluationReport(
            model_name="resnet18",
            checkpoint_name="resnet18_best",
            clean_accuracy=93.2,
            clean_loss=0.25,
            known_fault_accuracy=89.1,
            known_fault_drop=4.1,
            unseen_fault_accuracy=90.5,
            unseen_fault_drop=2.7,
            multi_fault_accuracies={2: 85.0, 3: 80.0},
            bit_flip_accuracies={"sign": 91.0},
            fgsm_accuracy=55.0,
            pgd_accuracy=42.0
        )
        d = rep.to_dict()
        assert d["model_name"] == "resnet18"
        assert d["dim1_clean"]["accuracy"] == 93.2
        assert d["dim6_adversarial"]["fgsm"] == 55.0
