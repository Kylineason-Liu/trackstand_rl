"""V1 steering position clipping and policy-rate target slew limiting."""

import math
from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils.configclass import configclass


class SlewLimitedJointPositionAction(JointPositionAction):
    """Scale/clip a position request, then slew once per 20-ms policy step.

    The inherited apply_actions() only writes the held target: eight 400-Hz
    physics applications do not advance the limiter eight times. Each reset
    primes only the resetting environments from their current joint positions,
    clipped to the command range, instead of retaining the last episode's target.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not math.isfinite(cfg.max_target_velocity_rad_s) or cfg.max_target_velocity_rad_s <= 0.0:
            raise ValueError("max_target_velocity_rad_s must be positive and finite")
        if not math.isclose(env.step_dt, 0.02, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("V1 steering slew updates require a 50-Hz (0.02-s) policy step")
        if cfg.clip is None:
            raise ValueError("V1 steering requires explicit position clip bounds")
        self._max_target_step_rad = cfg.max_target_velocity_rad_s * env.step_dt
        self._last_target = torch.zeros_like(self._processed_actions)
        self.reset()

    def process_actions(self, actions: torch.Tensor):
        # Stock preprocessing: raw * scale + offset, then cfg.clip.
        super().process_actions(actions)
        delta = (self._processed_actions - self._last_target).clamp(
            -self._max_target_step_rad, self._max_target_step_rad
        )
        self._last_target += delta
        self._processed_actions = self._last_target.clone()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        positions = self._asset.data.joint_pos.torch[:, self._joint_ids]
        self._last_target[ids] = torch.clamp(
            positions[ids], min=self._clip[ids, :, 0], max=self._clip[ids, :, 1]
        )
        self._processed_actions[ids] = self._last_target[ids]


@configclass
class SlewLimitedJointPositionActionCfg(JointPositionActionCfg):
    """The speed parameter limits target motion, not measured joint velocity."""

    class_type: type = SlewLimitedJointPositionAction
    max_target_velocity_rad_s: float = 23.9
