"""TD3 Agent: Complete Off-Policy Training Loop for Channel Discovery.

Implements Twin Delayed Deep Deterministic Policy Gradient (TD3):
  - Delayed actor updates (every `policy_delay` critic updates)
  - Target policy smoothing (Gaussian noise on target actions)
  - Clipped double Q-learning (min of twin critics for target)
  - Soft target network updates (Polyak averaging τ)

Reference: Fujimoto et al. (2018) "Addressing Function Approximation Error
           in Actor-Critic Methods" — arXiv:1802.09477
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from vulnshield.discovery.actor import TD3Actor
from vulnshield.discovery.critic import TD3TwinCritic
from vulnshield.discovery.replay_buffer import ReplayBuffer
from vulnshield.discovery.action_mapper import ActionMapper
from vulnshield.discovery.env import FaultDiscoveryEnv


@dataclass
class TD3Config:
    """Hyperparameters for TD3 training."""
    # Network
    hidden_dim: int = 256

    # Training
    gamma: float = 0.99          # Discount factor
    tau: float = 0.005           # Polyak averaging coefficient for target nets
    policy_delay: int = 2        # Actor update frequency (every N critic steps)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4

    # Exploration noise
    expl_noise_std: float = 0.1  # Std of Gaussian exploration noise (added to action)

    # Target policy smoothing
    target_noise_std: float = 0.2
    target_noise_clip: float = 0.5

    # Training logistics
    batch_size: int = 64
    replay_capacity: int = 10_000
    warmup_steps: int = 200      # Random actions before training starts

    # Checkpoint
    checkpoint_dir: str = "checkpoints/td3"


class TD3Agent:
    """Off-policy TD3 agent for vulnerable channel discovery.

    Args:
        obs_dim: Observation space dimension.
        action_dim: Action space dimension (always 2).
        config: Hyperparameter configuration.
        device: Compute device.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: Optional[TD3Config] = None,
        device: Optional[torch.device] = None
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config or TD3Config()
        self.device = device or torch.device("cpu")

        # Networks
        self.actor = TD3Actor(obs_dim, action_dim, self.config.hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = TD3TwinCritic(obs_dim, action_dim, self.config.hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        # Optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)

        # Replay buffer
        self.buffer = ReplayBuffer(
            obs_dim=obs_dim,
            action_dim=action_dim,
            capacity=self.config.replay_capacity,
            device=self.device
        )

        self._total_critic_updates = 0
        self.critic_losses: List[float] = []
        self.actor_losses: List[float] = []

    def select_action(
        self,
        state: torch.Tensor,
        explore: bool = True
    ) -> torch.Tensor:
        """Select action using the current actor + optional Gaussian exploration noise.

        Args:
            state: Current observation tensor (obs_dim,).
            explore: If True, adds Gaussian exploration noise.

        Returns:
            Clamped action tensor (action_dim,) in [-1, 1].
        """
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state.unsqueeze(0).to(self.device)).squeeze(0)
        self.actor.train()

        if explore:
            noise = torch.randn_like(action) * self.config.expl_noise_std
            action = (action + noise).clamp(-1.0, 1.0)

        return action

    def update(self) -> Dict[str, float]:
        """Perform one TD3 update step (critic + optional actor).

        Returns:
            Dict with 'critic_loss' and optionally 'actor_loss'.
        """
        if len(self.buffer) < self.config.batch_size:
            return {}

        cfg = self.config
        states, actions, rewards, next_states, dones = self.buffer.sample(cfg.batch_size)

        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(actions) * cfg.target_noise_std).clamp(
                -cfg.target_noise_clip, cfg.target_noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(-1.0, 1.0)

            # Twin Q targets
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target = rewards + (1.0 - dones) * cfg.gamma * torch.min(q1_target, q2_target)

        # Critic update
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
        self.critic_opt.step()

        self._total_critic_updates += 1
        self.critic_losses.append(critic_loss.item())
        metrics = {"critic_loss": critic_loss.item()}

        # Delayed actor update
        if self._total_critic_updates % cfg.policy_delay == 0:
            actor_loss = -self.critic.q1_value(states, self.actor(states)).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)
            self.actor_opt.step()

            # Soft update target networks (Polyak averaging)
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

            self.actor_losses.append(actor_loss.item())
            metrics["actor_loss"] = actor_loss.item()

        return metrics

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        tau = self.config.tau
        for src_p, tgt_p in zip(source.parameters(), target.parameters()):
            tgt_p.data.copy_(tau * src_p.data + (1.0 - tau) * tgt_p.data)

    def run_discovery(
        self,
        env: FaultDiscoveryEnv,
        max_total_queries: int = 50,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Run TD3 discovery under a strictly enforced total query budget.

        Mathematically guarantees query equivalence with heuristic baselines:
            Total Fault Injections performed == max_total_queries.

        Args:
            env: FaultDiscoveryEnv instance.
            max_total_queries: Absolute maximum number of fault evaluations allowed.
            checkpoint_dir: Directory to save TD3 checkpoints.
            verbose: Enable console progress reporting.

        Returns:
            Dict containing discovered channels and reward trajectory.
        """
        all_rewards: List[float] = []
        best_discoveries: List[Dict] = []
        step_count = 0
        ep = 0

        # Adjust warmup so it doesn't exceed 20% of query budget if budget is small
        warmup_limit = min(self.config.warmup_steps, int(max_total_queries * 0.2))

        if verbose:
            print(f"[*] TD3 Discovery: Strictly Enforced Global Budget = {max_total_queries} queries (Device: {self.device})")

        while step_count < max_total_queries:
            ep += 1
            state = env.reset()
            ep_reward = 0.0
            ep_discoveries: List[Dict] = []

            for _ in range(env.budget):
                if step_count >= max_total_queries:
                    break

                if step_count < warmup_limit:
                    # Uniform random exploration during warmup
                    action = torch.rand(self.action_dim) * 2.0 - 1.0
                else:
                    action = self.select_action(state, explore=True)

                result = env.step(action)
                step_count += 1

                self.buffer.push(state, action, result.reward, result.observation, result.done)
                self.update()

                state = result.observation
                ep_reward += result.reward

                if result.reward > 0:
                    ep_discoveries.append({
                        "layer_name": result.info["layer_name"],
                        "channel_idx": result.info["channel_idx"],
                        "delta_accuracy": result.info["delta_accuracy"],
                        "query_step": step_count
                    })

                if result.done:
                    break

            all_rewards.append(ep_reward)
            best_discoveries.extend(ep_discoveries)

            if verbose:
                print(f"  Ep {ep:2d} | Ep Reward: {ep_reward:6.2f} | Queries Used: {step_count}/{max_total_queries}")

        # Save checkpoint
        if checkpoint_dir is not None:
            save_dir = Path(checkpoint_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(self.actor.state_dict(), save_dir / "td3_actor.pt")
            torch.save(self.critic.state_dict(), save_dir / "td3_critic.pt")
            if verbose:
                print(f"[PASS] TD3 checkpoints saved to {save_dir}")

        # Rank discoveries by ΔA
        best_discoveries.sort(key=lambda d: d["delta_accuracy"], reverse=True)
        unique_channels: Dict[str, Dict] = {}
        for d in best_discoveries:
            key = f"{d['layer_name']}_c{d['channel_idx']}"
            if key not in unique_channels or d["delta_accuracy"] > unique_channels[key]["delta_accuracy"]:
                unique_channels[key] = d

        ranked = sorted(unique_channels.values(), key=lambda d: d["delta_accuracy"], reverse=True)

        return {
            "total_queries_executed": step_count,
            "max_budget_enforced": max_total_queries,
            "episode_rewards": all_rewards,
            "top_channels": ranked[:20],
            "total_episodes": ep
        }
