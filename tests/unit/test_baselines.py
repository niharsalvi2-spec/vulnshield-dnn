"""Unit Tests for Phase 7 — Baselines (Random, Activation, Gradient, DDPG)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from vulnshield.baselines.random_baseline import run_random_baseline
from vulnshield.baselines.activation_baseline import run_activation_baseline
from vulnshield.baselines.gradient_baseline import run_gradient_baseline
from vulnshield.baselines.ddpg_baseline import DDPGAgent, DDPGConfig, OUNoise
from vulnshield.models.resnet import resnet18


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny_model():
    m = resnet18(num_classes=10)
    m.eval()
    return m


@pytest.fixture(scope="module")
def calib_loader():
    images = torch.randn(32, 3, 32, 32)
    labels = torch.randint(0, 10, (32,))
    return DataLoader(TensorDataset(images, labels), batch_size=16, shuffle=False)


# ── Random Baseline ──────────────────────────────────────────────────────────

@pytest.mark.unit
class TestRandomBaseline:

    def test_returns_correct_budget(self, tiny_model, calib_loader):
        results = run_random_baseline(
            model=tiny_model,
            dataloader=calib_loader,
            clean_accuracy=10.0,
            budget=8,
            seed=42,
            device=torch.device("cpu")
        )
        assert len(results) == 8

    def test_results_contain_required_keys(self, tiny_model, calib_loader):
        results = run_random_baseline(
            model=tiny_model,
            dataloader=calib_loader,
            clean_accuracy=10.0,
            budget=4,
            seed=0,
            device=torch.device("cpu")
        )
        for r in results:
            assert "layer_name" in r
            assert "channel_idx" in r
            assert "fault_accuracy" in r
            assert "delta_accuracy" in r

    def test_results_sorted_descending(self, tiny_model, calib_loader):
        results = run_random_baseline(
            model=tiny_model,
            dataloader=calib_loader,
            clean_accuracy=10.0,
            budget=6,
            seed=7,
            device=torch.device("cpu")
        )
        deltas = [r["delta_accuracy"] for r in results]
        assert deltas == sorted(deltas, reverse=True)

    def test_delta_matches_clean_minus_fault(self, tiny_model, calib_loader):
        results = run_random_baseline(
            model=tiny_model,
            dataloader=calib_loader,
            clean_accuracy=10.0,
            budget=3,
            seed=99,
            device=torch.device("cpu")
        )
        for r in results:
            expected_delta = 10.0 - r["fault_accuracy"]
            assert abs(r["delta_accuracy"] - expected_delta) < 1e-4

    def test_deterministic_with_same_seed(self, tiny_model, calib_loader):
        r1 = run_random_baseline(tiny_model, calib_loader, 10.0, budget=4, seed=42, device=torch.device("cpu"))
        r2 = run_random_baseline(tiny_model, calib_loader, 10.0, budget=4, seed=42, device=torch.device("cpu"))
        assert [x["layer_name"] for x in r1] == [x["layer_name"] for x in r2]
        assert [x["channel_idx"] for x in r1] == [x["channel_idx"] for x in r2]


# ── Activation Baseline ──────────────────────────────────────────────────────

@pytest.mark.unit
class TestActivationBaseline:

    def test_returns_budget_results(self, tiny_model, calib_loader):
        results = run_activation_baseline(
            model=tiny_model,
            dataloader=calib_loader,
            budget=10,
            device=torch.device("cpu")
        )
        assert len(results) == 10

    def test_results_have_activation_score(self, tiny_model, calib_loader):
        results = run_activation_baseline(
            model=tiny_model,
            dataloader=calib_loader,
            budget=5,
            device=torch.device("cpu")
        )
        for r in results:
            assert "activation_score" in r
            assert r["activation_score"] >= 0.0

    def test_sorted_descending_by_score(self, tiny_model, calib_loader):
        results = run_activation_baseline(
            model=tiny_model,
            dataloader=calib_loader,
            budget=20,
            device=torch.device("cpu")
        )
        scores = [r["activation_score"] for r in results]
        assert scores == sorted(scores, reverse=True)


# ── Gradient / Taylor Baseline ───────────────────────────────────────────────

@pytest.mark.unit
class TestGradientBaseline:

    def test_returns_budget_results(self, tiny_model, calib_loader):
        results = run_gradient_baseline(
            model=tiny_model,
            dataloader=calib_loader,
            budget=10,
            device=torch.device("cpu")
        )
        assert len(results) == 10

    def test_results_have_gradient_score(self, tiny_model, calib_loader):
        results = run_gradient_baseline(
            model=tiny_model,
            dataloader=calib_loader,
            budget=5,
            device=torch.device("cpu")
        )
        for r in results:
            assert "gradient_score" in r
            assert r["gradient_score"] >= 0.0

    def test_sorted_descending_by_score(self, tiny_model, calib_loader):
        results = run_gradient_baseline(
            model=tiny_model,
            dataloader=calib_loader,
            budget=15,
            device=torch.device("cpu")
        )
        scores = [r["gradient_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_model_weights_unchanged_after_call(self, tiny_model, calib_loader):
        """Gradient baseline must not modify model parameters."""
        before = {n: p.data.clone() for n, p in tiny_model.named_parameters()}
        run_gradient_baseline(tiny_model, calib_loader, budget=4, device=torch.device("cpu"))
        for n, p in tiny_model.named_parameters():
            assert torch.equal(p.data, before[n]), f"Parameter {n} was modified!"


# ── OU Noise ─────────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestOUNoise:

    def test_sample_shape(self):
        noise = OUNoise(action_dim=2)
        s = noise.sample()
        assert s.shape == (2,)

    def test_reset_reinitialises_state(self):
        noise = OUNoise(action_dim=2, mu=0.0)
        noise.sample(); noise.sample()
        noise.reset()
        assert torch.allclose(noise.state, torch.zeros(2))


# ── DDPG Agent ───────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestDDPGAgent:

    def test_update_returns_losses(self):
        cfg = DDPGConfig(batch_size=8, warmup_steps=0, replay_capacity=100, hidden_dim=32)
        agent = DDPGAgent(obs_dim=4, action_dim=2, config=cfg, device=torch.device("cpu"))

        for _ in range(16):
            agent.buffer.push(
                torch.randn(4), torch.randn(2), 1.0, torch.randn(4), False
            )

        metrics = agent.update()
        assert "critic_loss" in metrics
        assert "actor_loss" in metrics
        assert metrics["critic_loss"] >= 0.0

    def test_select_action_shape(self):
        cfg = DDPGConfig(hidden_dim=32)
        agent = DDPGAgent(obs_dim=4, action_dim=2, config=cfg, device=torch.device("cpu"))
        action = agent.select_action(torch.randn(4), explore=False)
        assert action.shape == (2,)
        assert action.abs().max() <= 1.0 + 1e-5
