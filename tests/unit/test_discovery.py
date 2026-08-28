"""Unit Tests for Phase 6 — RL Discovery Agent (TD3)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from vulnshield.discovery.action_mapper import ActionMapper
from vulnshield.discovery.replay_buffer import ReplayBuffer
from vulnshield.discovery.actor import TD3Actor
from vulnshield.discovery.critic import TD3TwinCritic
from vulnshield.discovery.env import FaultDiscoveryEnv, OBS_DIM
from vulnshield.discovery.td3_agent import TD3Agent, TD3Config
from vulnshield.models.resnet import resnet18


# ── Fixtures ─────────────────────────────────────────────────────────────────

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


@pytest.fixture(scope="module")
def layer_channel_counts(tiny_model):
    from vulnshield.fault_injection.fault_injector import FaultInjector
    return FaultInjector(tiny_model).list_injectable_layers()


# ── ActionMapper ─────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestActionMapper:

    def test_decode_in_bounds(self, layer_channel_counts):
        mapper = ActionMapper(layer_channel_counts)
        for action in [
            torch.tensor([-1.0, -1.0]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([1.0, 1.0]),
            torch.tensor([0.5, -0.3])
        ]:
            layer_name, channel_idx = mapper.decode(action)
            _, out_channels = next(
                (n, c) for n, c in layer_channel_counts if n == layer_name
            )
            assert 0 <= channel_idx < out_channels

    def test_decode_extreme_actions_are_valid(self, layer_channel_counts):
        mapper = ActionMapper(layer_channel_counts)
        # These must never raise IndexError
        mapper.decode(torch.tensor([-1.0, -1.0]))
        mapper.decode(torch.tensor([1.0, 1.0]))
        mapper.decode(torch.tensor([1.1, -1.1]))   # out of range clamped

    def test_encode_decode_roundtrip(self, layer_channel_counts):
        mapper = ActionMapper(layer_channel_counts)
        for layer_idx in [0, 5, len(layer_channel_counts) - 1]:
            _, n_ch = layer_channel_counts[layer_idx]
            for channel_idx in [0, n_ch // 2, n_ch - 1]:
                action = mapper.encode(layer_idx, channel_idx)
                layer_name_out, channel_out = mapper.decode(action)
                layer_name_exp, _ = layer_channel_counts[layer_idx]
                assert layer_name_out == layer_name_exp
                assert channel_out == channel_idx

    def test_empty_layer_list_raises(self):
        with pytest.raises(ValueError):
            ActionMapper([])


# ── ReplayBuffer ─────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestReplayBuffer:

    def test_push_and_sample(self):
        buf = ReplayBuffer(obs_dim=4, action_dim=2, capacity=100)
        for _ in range(20):
            buf.push(
                torch.randn(4), torch.randn(2),
                float(torch.rand(1)), torch.randn(4), False
            )
        assert len(buf) == 20
        states, actions, rewards, next_states, dones = buf.sample(8)
        assert states.shape == (8, 4)
        assert actions.shape == (8, 2)
        assert rewards.shape == (8, 1)

    def test_buffer_circular_overwrite(self):
        buf = ReplayBuffer(obs_dim=4, action_dim=2, capacity=10)
        for _ in range(25):
            buf.push(torch.randn(4), torch.randn(2), 1.0, torch.randn(4), False)
        assert len(buf) == 10   # capped at capacity

    def test_sample_before_enough_data_raises(self):
        buf = ReplayBuffer(obs_dim=4, action_dim=2, capacity=100)
        buf.push(torch.randn(4), torch.randn(2), 1.0, torch.randn(4), False)
        with pytest.raises(RuntimeError):
            buf.sample(64)


# ── TD3 Networks ─────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestTD3Networks:

    def test_actor_forward_shape(self):
        actor = TD3Actor(obs_dim=4, action_dim=2)
        x = torch.randn(8, 4)
        out = actor(x)
        assert out.shape == (8, 2)
        assert (out.abs() <= 1.0 + 1e-5).all()   # Tanh output

    def test_actor_action_range(self):
        actor = TD3Actor(obs_dim=4, action_dim=2)
        for _ in range(10):
            out = actor(torch.randn(1, 4))
            assert out.min() >= -1.0 - 1e-5
            assert out.max() <= 1.0 + 1e-5

    def test_critic_forward_shape(self):
        critic = TD3TwinCritic(obs_dim=4, action_dim=2)
        s = torch.randn(8, 4)
        a = torch.randn(8, 2)
        q1, q2 = critic(s, a)
        assert q1.shape == (8, 1)
        assert q2.shape == (8, 1)

    def test_critic_q1_only(self):
        critic = TD3TwinCritic(obs_dim=4, action_dim=2)
        s, a = torch.randn(4, 4), torch.randn(4, 2)
        q1 = critic.q1_value(s, a)
        assert q1.shape == (4, 1)


# ── FaultDiscoveryEnv ────────────────────────────────────────────────────────

@pytest.mark.unit
class TestFaultDiscoveryEnv:

    def test_reset_returns_obs(self, tiny_model, eval_loader):
        env = FaultDiscoveryEnv(tiny_model, eval_loader, clean_accuracy=10.0, budget=3)
        obs = env.reset()
        assert obs.shape == (OBS_DIM,)
        assert torch.isfinite(obs).all()

    def test_step_returns_result(self, tiny_model, eval_loader):
        env = FaultDiscoveryEnv(tiny_model, eval_loader, clean_accuracy=10.0, budget=3)
        env.reset()
        action = torch.tensor([0.0, 0.0])
        result = env.step(action)
        assert result.observation.shape == (OBS_DIM,)
        assert isinstance(result.reward, float)
        assert isinstance(result.done, bool)
        assert "layer_name" in result.info

    def test_episode_terminates_at_budget(self, tiny_model, eval_loader):
        budget = 3
        env = FaultDiscoveryEnv(tiny_model, eval_loader, clean_accuracy=10.0, budget=budget)
        env.reset()
        done = False
        steps = 0
        while not done:
            result = env.step(torch.rand(2) * 2 - 1)
            done = result.done
            steps += 1
        assert steps == budget


# ── TD3Agent update loop ─────────────────────────────────────────────────────

@pytest.mark.unit
class TestTD3AgentUpdate:

    def test_update_returns_losses_after_warmup(self):
        cfg = TD3Config(
            batch_size=8,
            warmup_steps=0,
            replay_capacity=200,
            hidden_dim=32
        )
        agent = TD3Agent(obs_dim=4, action_dim=2, config=cfg, device=torch.device("cpu"))

        # Pre-fill buffer
        for _ in range(16):
            s = torch.randn(4)
            a = torch.randn(2)
            ns = torch.randn(4)
            agent.buffer.push(s, a, 1.0, ns, False)

        metrics = agent.update()
        assert "critic_loss" in metrics
        assert metrics["critic_loss"] >= 0.0

    def test_run_discovery_strictly_obeys_budget(self, tiny_model, eval_loader):
        env = FaultDiscoveryEnv(tiny_model, eval_loader, clean_accuracy=10.0, budget=10)
        cfg = TD3Config(hidden_dim=32, warmup_steps=2)
        agent = TD3Agent(obs_dim=env.obs_dim, action_dim=env.action_dim, config=cfg, device=torch.device("cpu"))

        # Strict budget bound: exactly 6 queries
        res = agent.run_discovery(env, max_total_queries=6, verbose=False)
        assert res["total_queries_executed"] == 6
        assert res["max_budget_enforced"] == 6
