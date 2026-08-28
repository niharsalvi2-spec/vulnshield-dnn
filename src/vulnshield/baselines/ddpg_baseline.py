"""Layer-wise DDPG Baseline Agent.

Simplified single-critic DDPG (no twin critics, no delayed actor updates,
no target policy smoothing) operating independently per layer.

Architectural differences from TD3:
  - Single Q-network (not twin)
  - Actor updated every critic update (no delay)
  - No target policy smoothing noise
  - Ornstein-Uhlenbeck (OU) exploration noise

This serves as the RL baseline to isolate the contribution of TD3-specific
improvements (twin critics, delayed updates, target smoothing) over basic DDPG.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from vulnshield.discovery.actor import TD3Actor
from vulnshield.discovery.replay_buffer import ReplayBuffer
from vulnshield.discovery.env import FaultDiscoveryEnv


class _SingleCriticDDPG(nn.Module):
    """Single Q-function for DDPG (contrast with TD3's twin critics)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


class OUNoise:
    """Ornstein-Uhlenbeck noise for temporally correlated exploration."""

    def __init__(self, action_dim: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = torch.ones(self.action_dim) * self.mu

    def sample(self) -> torch.Tensor:
        dx = self.theta * (self.mu - self.state) + self.sigma * torch.randn(self.action_dim)
        self.state = self.state + dx
        return self.state.clone()


@dataclass
class DDPGConfig:
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    ou_theta: float = 0.15
    ou_sigma: float = 0.2
    batch_size: int = 64
    replay_capacity: int = 10_000
    warmup_steps: int = 200


class DDPGAgent:
    """Layer-wise DDPG baseline agent (single critic, no TD3 tricks)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: Optional[DDPGConfig] = None,
        device: Optional[torch.device] = None
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config or DDPGConfig()
        self.device = device or torch.device("cpu")

        self.actor = TD3Actor(obs_dim, action_dim, self.config.hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = _SingleCriticDDPG(obs_dim, action_dim, self.config.hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)

        self.buffer = ReplayBuffer(
            obs_dim=obs_dim,
            action_dim=action_dim,
            capacity=self.config.replay_capacity,
            device=self.device
        )
        self.noise = OUNoise(action_dim, theta=self.config.ou_theta, sigma=self.config.ou_sigma)

    def select_action(self, state: torch.Tensor, explore: bool = True) -> torch.Tensor:
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state.unsqueeze(0).to(self.device)).squeeze(0)
        self.actor.train()
        if explore:
            action = (action + self.noise.sample().to(self.device)).clamp(-1.0, 1.0)
        return action

    def update(self) -> Dict[str, float]:
        if len(self.buffer) < self.config.batch_size:
            return {}
        cfg = self.config
        states, actions, rewards, next_states, dones = self.buffer.sample(cfg.batch_size)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_target = rewards + (1.0 - dones) * cfg.gamma * self.critic_target(next_states, next_actions)

        # Critic update (single Q)
        q_pred = self.critic(states, actions)
        critic_loss = F.mse_loss(q_pred, q_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_opt.step()

        # Actor update every step (no delay — DDPG style)
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
        self.actor_opt.step()

        # Soft target updates
        for src, tgt in [(self.actor, self.actor_target), (self.critic, self.critic_target)]:
            for p, tp in zip(src.parameters(), tgt.parameters()):
                tp.data.copy_(cfg.tau * p.data + (1.0 - cfg.tau) * tp.data)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}

    def run_discovery(
        self,
        env: FaultDiscoveryEnv,
        num_episodes: int = 20
    ) -> Dict[str, Any]:
        all_rewards: List[float] = []
        best_discoveries: List[Dict] = []
        step_count = 0

        print(f"[*] DDPG Discovery: {num_episodes} episodes, budget={env.budget}")

        for ep in range(1, num_episodes + 1):
            state = env.reset()
            self.noise.reset()
            ep_reward = 0.0

            for _ in range(env.budget):
                if step_count < self.config.warmup_steps:
                    action = torch.rand(self.action_dim) * 2.0 - 1.0
                else:
                    action = self.select_action(state, explore=True)

                result = env.step(action)
                self.buffer.push(state, action, result.reward, result.observation, result.done)
                self.update()

                state = result.observation
                ep_reward += result.reward
                step_count += 1

                if result.reward > 0:
                    best_discoveries.append({
                        "layer_name": result.info["layer_name"],
                        "channel_idx": result.info["channel_idx"],
                        "delta_accuracy": result.info["delta_accuracy"]
                    })
                if result.done:
                    break

            all_rewards.append(ep_reward)
            avg = sum(all_rewards[-5:]) / min(len(all_rewards), 5)
            print(f"  Episode {ep:3d}/{num_episodes} | Ep Reward: {ep_reward:.3f} | Avg(5): {avg:.3f}")

        best_discoveries.sort(key=lambda d: d["delta_accuracy"], reverse=True)
        unique: Dict[str, Dict] = {}
        for d in best_discoveries:
            k = f"{d['layer_name']}_c{d['channel_idx']}"
            if k not in unique or d["delta_accuracy"] > unique[k]["delta_accuracy"]:
                unique[k] = d

        return {
            "episode_rewards": all_rewards,
            "top_channels": sorted(unique.values(), key=lambda d: d["delta_accuracy"], reverse=True)[:20],
            "total_steps": step_count
        }
