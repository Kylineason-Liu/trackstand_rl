"""External velocity integral, supplied as feedforward to an unchanged IdealPDActuator."""

import math
from collections.abc import Sequence

import torch

from isaaclab.actuators import IdealPDActuator
from isaaclab.envs.mdp.actions import JointVelocityAction, JointVelocityActionCfg
from isaaclab.utils.configclass import configclass


class VelocityIntegral:
    """Per-environment trapezoidal I state with conditional-integration anti-windup.

    The first sample after reset only primes the integrator. Target changes do
    not reset it. As in the hardware test, even rejected increments advance the
    previous-error sample, and clamping corrects the stored state itself.
    """

    def __init__(self, template: torch.Tensor, gain_nm_per_rad: float):
        if not math.isfinite(gain_nm_per_rad) or gain_nm_per_rad < 0.0:
            raise ValueError("integral_gain_nm_per_rad must be finite and nonnegative")
        self.gain_nm_per_rad = gain_nm_per_rad
        self.error_rad = torch.zeros_like(template)
        self.torque_nm = torch.zeros_like(template)
        self.previous_error = torch.zeros_like(template)
        self.initialized = torch.zeros_like(template, dtype=torch.bool)
        self.limited_updates = torch.zeros(template.shape[0], device=template.device, dtype=torch.long)

    def update(self, error: torch.Tensor, dt: float, proportional_gain: torch.Tensor, effort_limit: torch.Tensor):
        if self.gain_nm_per_rad == 0.0:
            return
        increment = torch.where(self.initialized, 0.5 * (error + self.previous_error) * dt, 0.0)
        proposed = self.gain_nm_per_rad * (self.error_rad + increment)
        candidate = proposed.clamp(min=-effort_limit, max=effort_limit)
        delta = candidate - self.torque_nm
        total = proportional_gain * error + candidate
        freeze = ((total > effort_limit) & (delta > 0)) | ((total < -effort_limit) & (delta < 0))
        candidate = torch.where(freeze, self.torque_nm, candidate)
        self.limited_updates += (candidate != proposed).any(dim=-1)
        self.torque_nm.copy_(candidate)
        self.error_rad.copy_(candidate / self.gain_nm_per_rad)
        self.previous_error.copy_(error)
        self.initialized.fill_(True)

    def reset(self, env_ids: Sequence[int] | None = None):
        ids = slice(None) if env_ids is None else env_ids
        self.error_rad[ids] = 0.0
        self.torque_nm[ids] = 0.0
        self.previous_error[ids] = 0.0
        self.initialized[ids] = False
        self.limited_updates[ids] = 0


class IntegralJointVelocityAction(JointVelocityAction):
    """Hold 50-Hz velocity targets; update external I independently at 200 Hz.

    IdealPD still evaluates its P term and clips P+I at every physics step.
    This action owns the effort target of its configured actuator; no separate
    joint-effort action may write those same joints.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not math.isfinite(cfg.integral_rate_hz) or cfg.integral_rate_hz <= 0.0:
            raise ValueError("integral_rate_hz must be positive and finite")
        ratio = 1.0 / (cfg.integral_rate_hz * env.physics_dt)
        if ratio < 1.0 or not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("Physics frequency must be an integer multiple of integral_rate_hz")
        if getattr(env, "_physics_handles_decimation", False):
            raise ValueError("Velocity PI requires action application at each physics step")
        self._integral_decimation = round(ratio)
        self._integral_dt = self._integral_decimation * env.physics_dt
        self._physics_steps = 0
        self._actuator = self._asset.actuators[cfg.actuator_name]
        if type(self._actuator) is not IdealPDActuator or self._actuator.joint_names != self._joint_names:
            raise ValueError("Velocity PI requires an IdealPDActuator with exactly the action's joints")
        if torch.any(self._actuator.stiffness != 0.0):
            raise ValueError("Velocity PI requires zero position stiffness")
        self.integral = VelocityIntegral(self._processed_actions, cfg.integral_gain_nm_per_rad)

    def apply_actions(self):
        if self._physics_steps % self._integral_decimation == 0:
            speed = self._asset.data.joint_vel.torch[:, self._joint_ids]
            self.integral.update(
                self.processed_actions - speed,
                self._integral_dt,
                self._actuator.damping,
                self._actuator.effort_limit,
            )
        self._physics_steps += 1
        super().apply_actions()
        self._asset.set_joint_effort_target_index(target=self.integral.torque_nm, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.integral.reset(env_ids)
        self._processed_actions[ids] = 0.0
        # Manager resets precede reset-time scene writes. Clear both targets now
        # so an autoreset cannot submit the preceding episode's retained torque.
        self._asset.set_joint_velocity_target_index(
            target=self._processed_actions[ids], joint_ids=self._joint_ids, env_ids=env_ids
        )
        self._asset.set_joint_effort_target_index(
            target=self.integral.torque_nm[ids], joint_ids=self._joint_ids, env_ids=env_ids
        )


@configclass
class IntegralJointVelocityActionCfg(JointVelocityActionCfg):
    """I gain is torque per integrated velocity error; controller P lives in IdealPD damping."""

    class_type: type = IntegralJointVelocityAction
    actuator_name: str = "drive"
    integral_gain_nm_per_rad: float = 3.3
    integral_rate_hz: float = 200.0
