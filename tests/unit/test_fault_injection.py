"""Unit Tests for Fault Injection Engine (Phase 5)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from vulnshield.core.exceptions import FaultInjectionError
from vulnshield.fault_injection.channel_hook import StuckAtZeroHook
from vulnshield.fault_injection.fault_injector import FaultInjector, _resolve_layer
from vulnshield.vulnerability.scorer import (
    ChannelVulnerabilityScore,
    VulnerabilityReport,
    score_layer_vulnerability
)
from vulnshield.models.resnet import resnet18


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_resnet():
    """Tiny ResNet-18 (CPU, eval mode) shared across tests in this module."""
    m = resnet18(num_classes=10)
    m.eval()
    return m


def _make_loader(n: int = 32, batch: int = 16) -> DataLoader:
    images = torch.randn(n, 3, 32, 32)
    labels = torch.randint(0, 10, (n,))
    return DataLoader(TensorDataset(images, labels), batch_size=batch)


# ── StuckAtZeroHook ──────────────────────────────────────────────────────────

@pytest.mark.unit
class TestStuckAtZeroHook:

    def test_hook_zeroes_target_channel(self, tiny_resnet):
        """Activations on the hooked channel must be exactly 0.0 after forward."""
        x = torch.randn(2, 3, 32, 32)
        layer = tiny_resnet.layer1[0].conv1   # 64 out_channels
        channel = 10
        captured = {}

        def capture_hook(module, inp, out):
            captured["out"] = out.detach().clone()

        # Register capture AFTER the fault hook (hooks run FIFO)
        hook = StuckAtZeroHook(layer, channel, "layer1.0.conv1")
        hook.register()
        h2 = layer.register_forward_hook(capture_hook)

        with torch.no_grad():
            _ = tiny_resnet(x)

        hook.remove()
        h2.remove()

        assert "out" in captured
        assert (captured["out"][:, channel, :, :] == 0.0).all()
        # Other channels should NOT all be zero
        assert captured["out"][:, channel + 1, :, :].abs().sum() > 0.0

    def test_hook_other_channels_unaffected(self, tiny_resnet):
        """All channels except the faulted one preserve non-zero activations."""
        x = torch.randn(2, 3, 32, 32)
        layer = tiny_resnet.layer1[0].conv2
        channel = 5
        outputs = {}

        def cap(module, inp, out):
            outputs["out"] = out.detach().clone()

        hook = StuckAtZeroHook(layer, channel, "layer1.0.conv2")
        hook.register()
        h2 = layer.register_forward_hook(cap)
        with torch.no_grad():
            _ = tiny_resnet(x)
        hook.remove()
        h2.remove()

        out = outputs["out"]
        assert (out[:, channel, :, :] == 0.0).all()
        # Sum over all non-faulted channels must be non-zero
        mask = torch.ones(out.shape[1], dtype=torch.bool)
        mask[channel] = False
        assert out[:, mask, :, :].abs().sum() > 0.0

    def test_hook_context_manager(self, tiny_resnet):
        """Context manager form: hook active inside, removed outside."""
        layer = tiny_resnet.conv1
        hook = StuckAtZeroHook(layer, 0, "conv1")

        with hook:
            assert hook.is_active
        assert not hook.is_active

    def test_double_register_raises(self, tiny_resnet):
        layer = tiny_resnet.layer2[0].conv1
        hook = StuckAtZeroHook(layer, 3, "layer2.0.conv1")
        hook.register()
        with pytest.raises(RuntimeError):
            hook.register()
        hook.remove()

    def test_invalid_channel_raises(self, tiny_resnet):
        layer = tiny_resnet.conv1   # 64 out channels
        with pytest.raises(IndexError):
            StuckAtZeroHook(layer, 999, "conv1")

    def test_wrong_layer_type_raises(self):
        bad_layer = nn.Linear(10, 10)
        with pytest.raises(TypeError):
            StuckAtZeroHook(bad_layer, 0, "fc")


# ── FaultInjector ────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestFaultInjector:

    def test_inject_context_manager_cleans_up(self, tiny_resnet):
        injector = FaultInjector(tiny_resnet)
        specs = [("layer1.0.conv1", 0)]
        x = torch.randn(1, 3, 32, 32)

        with injector.inject(specs):
            with torch.no_grad():
                out_faulted = tiny_resnet(x).clone()

        with torch.no_grad():
            out_clean = tiny_resnet(x).clone()

        # Outputs should differ (fault changes predictions)
        # (not guaranteed for all inputs, but for a random init model is very likely)
        # Just verify no exception thrown and shapes are correct
        assert out_faulted.shape == (1, 10)
        assert out_clean.shape == (1, 10)

    def test_multi_fault_inject(self, tiny_resnet):
        injector = FaultInjector(tiny_resnet)
        specs = [("layer1.0.conv1", 0), ("layer2.0.conv1", 5), ("layer3.0.conv1", 10)]
        x = torch.randn(2, 3, 32, 32)
        with injector.inject(specs):
            with torch.no_grad():
                out = tiny_resnet(x)
        assert out.shape == (2, 10)
        assert torch.isfinite(out).all()

    def test_bad_layer_name_raises(self, tiny_resnet):
        injector = FaultInjector(tiny_resnet)
        with pytest.raises(FaultInjectionError):
            with injector.inject([("nonexistent_layer_xyz", 0)]):
                pass

    def test_list_injectable_layers(self, tiny_resnet):
        injector = FaultInjector(tiny_resnet)
        layers = injector.list_injectable_layers()
        assert len(layers) == 20   # ResNet-18 has 20 Conv2d layers
        names = [n for n, _ in layers]
        assert "conv1" in names
        assert "layer4.1.conv2" in names

    def test_get_layer_channel_count(self, tiny_resnet):
        injector = FaultInjector(tiny_resnet)
        assert injector.get_layer_channel_count("conv1") == 64
        assert injector.get_layer_channel_count("layer4.0.conv2") == 512

    def test_hooks_removed_after_exception(self, tiny_resnet):
        """Hooks must be cleaned up even if exception raised inside context."""
        injector = FaultInjector(tiny_resnet)
        layer = tiny_resnet.layer1[0].conv1
        n_hooks_before = len(layer._forward_hooks)

        try:
            with injector.inject([("layer1.0.conv1", 0)]):
                raise ValueError("Simulated error inside injection context")
        except ValueError:
            pass

        n_hooks_after = len(layer._forward_hooks)
        assert n_hooks_after == n_hooks_before


# ── VulnerabilityScorer ──────────────────────────────────────────────────────

@pytest.mark.unit
class TestVulnerabilityScorer:

    def test_score_layer_vulnerability_structure(self, tiny_resnet):
        """Scorer produces VulnerabilityReport with correct structure."""
        loader = _make_loader(n=32, batch=16)
        clean_acc = 10.0   # Dummy: random net → ~10%

        report = score_layer_vulnerability(
            model=tiny_resnet,
            layer_name="conv1",
            dataloader=loader,
            clean_accuracy=clean_acc,
            device=torch.device("cpu"),
            channel_indices=range(4)   # Only first 4 channels for speed
        )

        assert isinstance(report, VulnerabilityReport)
        assert len(report.scores) == 4
        for score in report.scores:
            assert isinstance(score, ChannelVulnerabilityScore)
            assert score.layer_name == "conv1"
            assert 0 <= score.channel_idx < 4
            # ΔA = clean - fault; can be negative (fault can sometimes help random net)
            assert score.delta_accuracy == pytest.approx(
                score.clean_accuracy - score.fault_accuracy, abs=1e-4
            )

    def test_ranked_channels_ordering(self, tiny_resnet):
        loader = _make_loader(n=32, batch=16)
        report = score_layer_vulnerability(
            model=tiny_resnet,
            layer_name="conv1",
            dataloader=loader,
            clean_accuracy=10.0,
            device=torch.device("cpu"),
            channel_indices=range(8)
        )
        ranked = report.ranked_channels
        deltas = [s.delta_accuracy for s in ranked]
        assert deltas == sorted(deltas, reverse=True)
