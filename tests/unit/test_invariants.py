"""Unit Tests for Scientific Invariants and Data Leakage Audits."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from vulnshield.models.resnet import resnet18
from vulnshield.fault_injection.fault_injector import FaultInjector
from vulnshield.discovery.action_mapper import ActionMapper
from vulnshield.validation.invariants import (
    verify_weight_immutability,
    verify_bit_flip_reversibility,
    verify_action_mapper_soundness
)
from vulnshield.validation.leakage_audit import DataLeakageAuditor


@pytest.mark.unit
class TestScientificInvariants:

    def test_weight_immutability_during_fault_injection(self):
        model = resnet18(num_classes=10)
        injector = FaultInjector(model)
        dummy_input = torch.randn(2, 3, 32, 32)

        is_immutable = verify_weight_immutability(model, injector, dummy_input)
        assert is_immutable is True

    def test_bit_flip_reversibility(self):
        test_t = torch.tensor([1.234, -5.678, 0.001], dtype=torch.float32)
        for bit_pos in [0, 10, 23, 27, 31]:
            assert verify_bit_flip_reversibility(test_t, bit_pos) is True

    def test_action_mapper_soundness(self):
        model = resnet18(num_classes=10)
        counts = FaultInjector(model).list_injectable_layers()
        mapper = ActionMapper(counts)

        res = verify_action_mapper_soundness(mapper, grid_resolution=60)
        assert res["all_in_bounds"] is True
        assert res["all_layers_reachable"] is True

    def test_data_leakage_auditor_zero_leakage(self):
        train_idx = list(range(0, 40000))
        val_idx = list(range(40000, 45000))
        test_idx = list(range(45000, 50000))
        fault_idx = list(range(40000, 41000))  # Disjoint from test_idx

        audit = DataLeakageAuditor.audit_splits(train_idx, val_idx, test_idx, fault_idx)
        assert audit["has_leakage"] is False
        assert audit["overlaps"]["train_test_overlap"] == 0
        assert audit["overlaps"]["val_test_overlap"] == 0
        assert audit["overlaps"]["test_fault_overlap"] == 0
