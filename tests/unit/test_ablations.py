"""Unit Tests for TD3 Scientific Component Ablation Suite."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from vulnshield.models.resnet import resnet18
from vulnshield.discovery.ablations import (
    ABLATION_SPECS,
    run_single_ablation,
    run_full_ablation_suite
)


@pytest.fixture(scope="module")
def tiny_model():
    m = resnet18(num_classes=10)
    m.eval()
    return m


@pytest.fixture(scope="module")
def eval_loader():
    images = torch.randn(16, 3, 32, 32)
    labels = torch.randint(0, 10, (16,))
    return DataLoader(TensorDataset(images, labels), batch_size=8)


@pytest.mark.unit
class TestAblationSuite:

    def test_ablation_specs_count(self):
        assert len(ABLATION_SPECS) == 6
        ids = [s.ablation_id for s in ABLATION_SPECS]
        assert ids == ["A0", "A1", "A2", "A3", "A4", "A5"]

    def test_run_single_ablation_random(self, tiny_model, eval_loader):
        spec = ABLATION_SPECS[0]  # A0
        res = run_single_ablation(spec, tiny_model, eval_loader, clean_accuracy=10.0, max_total_queries=4, seed=42)
        assert res["ablation_id"] == "A0"
        assert res["queries_executed"] == 4

    def test_run_single_ablation_td3_no_twin(self, tiny_model, eval_loader):
        spec = ABLATION_SPECS[2]  # A2: no twin Q
        res = run_single_ablation(spec, tiny_model, eval_loader, clean_accuracy=10.0, max_total_queries=4, seed=42)
        assert res["ablation_id"] == "A2"
        assert res["queries_executed"] == 4
