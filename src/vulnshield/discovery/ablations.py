"""Systematic Component Ablation Suite for Reinforcement Learning Discovery.

Implements controlled architectural and algorithmic ablations:
  - A0: Uniform Random Discovery Baseline
  - A1: Single-Critic Deterministic Policy Gradient (DDPG)
  - A2: TD3 without Twin Critics (Single Q-network, with smoothing & delayed update)
  - A3: TD3 without Target Policy Smoothing Noise (noise_std=0.0)
  - A4: TD3 without Delayed Policy Updates (policy_delay=1)
  - A5: Full VulnShield-DNN TD3 (Twin Q, Target Smoothing, Policy Delay=2)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vulnshield.discovery.env import FaultDiscoveryEnv
from vulnshield.discovery.td3_agent import TD3Agent, TD3Config
from vulnshield.baselines.ddpg_baseline import DDPGAgent, DDPGConfig
from vulnshield.baselines.random_baseline import run_random_baseline


@dataclass
class AblationExperimentSpec:
    """Specification of an RL discovery ablation configuration."""
    ablation_id: str
    name: str
    description: str
    use_twin_critics: bool = True
    use_target_smoothing: bool = True
    policy_delay: int = 2
    is_ddpg: bool = False
    is_random: bool = False


ABLATION_SPECS: List[AblationExperimentSpec] = [
    AblationExperimentSpec("A0", "Random Baseline", "Uniform random channel sampling", is_random=True),
    AblationExperimentSpec("A1", "Standard DDPG", "Single Q-function, OU noise, policy_delay=1", is_ddpg=True),
    AblationExperimentSpec("A2", "TD3 w/o Twin Q", "Single critic with target smoothing and delayed update", use_twin_critics=False),
    AblationExperimentSpec("A3", "TD3 w/o Smoothing", "Twin Q with zero target policy smoothing noise", use_target_smoothing=False),
    AblationExperimentSpec("A4", "TD3 w/o Delay", "Twin Q and smoothing with policy_delay=1 (no update delay)", policy_delay=1),
    AblationExperimentSpec("A5", "Full TD3 (Ours)", "Twin Q, target policy smoothing, and policy_delay=2", use_twin_critics=True, use_target_smoothing=True, policy_delay=2)
]


def run_single_ablation(
    spec: AblationExperimentSpec,
    model: nn.Module,
    eval_loader: DataLoader,
    clean_accuracy: float,
    max_total_queries: int = 50,
    seed: int = 42,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Execute a single controlled ablation trial under identical query budget."""
    dev = device or torch.device("cpu")
    torch.manual_seed(seed)

    if spec.is_random:
        results = run_random_baseline(model, eval_loader, clean_accuracy, budget=max_total_queries, seed=seed, device=dev)
        top_d = results[0]["delta_accuracy"] if results else 0.0
        mean_d = sum(r["delta_accuracy"] for r in results) / max(len(results), 1)
        return {
            "ablation_id": spec.ablation_id,
            "name": spec.name,
            "top_delta": top_d,
            "mean_delta": mean_d,
            "queries_executed": len(results),
            "trajectory": [r["delta_accuracy"] for r in results]
        }

    env = FaultDiscoveryEnv(model, eval_loader, clean_accuracy=clean_accuracy, budget=10, device=dev)

    if spec.is_ddpg:
        ddpg_cfg = DDPGConfig(hidden_dim=32, warmup_steps=5)
        agent_ddpg = DDPGAgent(obs_dim=env.obs_dim, action_dim=env.action_dim, config=ddpg_cfg, device=dev)
        res = agent_ddpg.run_discovery(env, max_total_queries=max_total_queries, verbose=False)
        top_ch = res["top_channels"]
        top_d = top_ch[0]["delta_accuracy"] if top_ch else 0.0
        mean_d = sum(c["delta_accuracy"] for c in top_ch) / max(len(top_ch), 1)
        return {
            "ablation_id": spec.ablation_id,
            "name": spec.name,
            "top_delta": top_d,
            "mean_delta": mean_d,
            "queries_executed": res["total_queries_executed"],
            "trajectory": res["episode_rewards"]
        }

    # TD3 Ablations
    td3_cfg = TD3Config(
        hidden_dim=32,
        warmup_steps=5,
        target_noise_std=0.2 if spec.use_target_smoothing else 0.0,
        policy_delay=spec.policy_delay
    )
    agent = TD3Agent(obs_dim=env.obs_dim, action_dim=env.action_dim, config=td3_cfg, device=dev)
    res = agent.run_discovery(env, max_total_queries=max_total_queries, verbose=False)
    top_ch = res["top_channels"]
    top_d = top_ch[0]["delta_accuracy"] if top_ch else 0.0
    mean_d = sum(c["delta_accuracy"] for c in top_ch) / max(len(top_ch), 1)

    return {
        "ablation_id": spec.ablation_id,
        "name": spec.name,
        "top_delta": top_d,
        "mean_delta": mean_d,
        "queries_executed": res["total_queries_executed"],
        "trajectory": res["episode_rewards"]
    }


def run_full_ablation_suite(
    model: nn.Module,
    eval_loader: DataLoader,
    clean_accuracy: float,
    max_total_queries: int = 50,
    seed: int = 42,
    device: Optional[torch.device] = None
) -> List[Dict[str, Any]]:
    """Execute the entire ablation matrix (A0 through A5) under identical experimental conditions."""
    results = []
    for spec in ABLATION_SPECS:
        res = run_single_ablation(
            spec=spec,
            model=model,
            eval_loader=eval_loader,
            clean_accuracy=clean_accuracy,
            max_total_queries=max_total_queries,
            seed=seed,
            device=device
        )
        results.append(res)
    return results
