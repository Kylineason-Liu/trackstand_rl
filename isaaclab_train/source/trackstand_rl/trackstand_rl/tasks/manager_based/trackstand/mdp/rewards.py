"""Rewards used by both stationary and dynamic balancing."""

import torch
from isaaclab.envs import ManagerBasedRLEnv


def action_rate_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize rate changes of the full action vector."""
    return -torch.norm(env.action_manager.action - env.action_manager.prev_action, p=2, dim=-1)


def __getattr__(name: str):
    """Resolve locomotion-only rewards from ``mdp.dynamic.rewards`` (deploy yaml)."""
    from .dynamic import rewards as _dynamic_rewards

    try:
        return getattr(_dynamic_rewards, name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
