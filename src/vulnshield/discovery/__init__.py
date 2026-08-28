"""VulnShield-DNN Discovery Package — TD3 Channel Vulnerability Discovery."""

from vulnshield.discovery.action_mapper import ActionMapper
from vulnshield.discovery.replay_buffer import ReplayBuffer
from vulnshield.discovery.actor import TD3Actor
from vulnshield.discovery.critic import TD3TwinCritic
from vulnshield.discovery.env import FaultDiscoveryEnv, OBS_DIM
from vulnshield.discovery.td3_agent import TD3Agent, TD3Config

__all__ = [
    "ActionMapper",
    "ReplayBuffer",
    "TD3Actor",
    "TD3TwinCritic",
    "FaultDiscoveryEnv",
    "OBS_DIM",
    "TD3Agent",
    "TD3Config"
]
