# Ported from ~/bike_test/CycleRL/source/CycleRL/CycleRL/tasks/manager_based/cyclerl/mdp/dynamic/actions.py (imports/paths rewritten for trackstand_rl locomotive).
from __future__ import annotations

from collections.abc import Sequence

import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg, JointVelocityActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction, JointVelocityAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers import ActionTermCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils.buffers.delay_buffer import DelayBuffer
from isaaclab.utils.configclass import configclass

# ======================================================================
# 1. TENSORIZED PID CONTROLLER
# ======================================================================
class TensorPID:
    """A PyTorch-native PID controller for massively parallel environments."""
    def __init__(self, kp, ki, kd, num_envs, device, out_limit=None, i_limit=5.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.out_limit = out_limit
        self.i_limit = i_limit
        
        self.integral = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.prev_error = torch.zeros(num_envs, dtype=torch.float32, device=device)
        
    def reset(self, env_ids=None):
        if env_ids is None:
            self.integral.zero_()
            self.prev_error.zero_()
        else:
            self.integral[env_ids] = 0.0
            self.prev_error[env_ids] = 0.0
            
    def step(self, setpoint, measured, measured_rate, dt):
        error = setpoint - measured
        self.integral += error * dt
        self.integral = torch.clamp(self.integral, -self.i_limit, self.i_limit)
        
        out = (self.kp * error) + (self.ki * self.integral) - (self.kd * measured_rate)
        
        if self.out_limit is not None:
            out = torch.clamp(out, -self.out_limit, self.out_limit)
            
        self.prev_error = error
        return out

# ======================================================================
# 2. AUTONOMOUS ACTION TERM IMPLEMENTATION
# ======================================================================
class CascadeFlywheelAction(ActionTerm):
    # Because of __future__ annotations, Python reads this without crashing!
    cfg: CascadeFlywheelActionCfg
    
    def __init__(self, cfg: CascadeFlywheelActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        
        # Resolve joint index from string name
        joint_ids, _ = self.robot.find_joints(cfg.joint_name)
        self.joint_idx = joint_ids[0]
        self.flywheel_idx = self.joint_idx  # Alias for diagnostics
        
        # Pre-allocate joint index tensor for Warp specifically as int32!
        self.joint_idx_tensor = torch.tensor([self.joint_idx], dtype=torch.int32, device=env.device)

        # Resolve additional joints for diagnostics
        steer_ids, _ = self.robot.find_joints("Revolute_7")
        self.steering_idx = steer_ids[0] if len(steer_ids) > 0 else 0
        
        drive_ids, _ = self.robot.find_joints("Revolute_9")
        self.drive_idx = drive_ids[0] if len(drive_ids) > 0 else 0
        
        # Initialize the 3 Cascade PIDs
        self.pid_vel = TensorPID(cfg.vel_kp, cfg.vel_ki, cfg.vel_kd, env.num_envs, env.device, cfg.vel_out_limit, cfg.vel_i_limit)
        self.pid_angle = TensorPID(cfg.angle_kp, cfg.angle_ki, cfg.angle_kd, env.num_envs, env.device, cfg.angle_out_limit, cfg.angle_i_limit)
        self.pid_gyro = TensorPID(cfg.gyro_kp, cfg.gyro_ki, cfg.gyro_kd, env.num_envs, env.device, cfg.gyro_out_limit, cfg.gyro_i_limit)
        
        # State tracking
        self.prev_lean = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        
        # Setting action_dim to 0 tells IsaacLab the RL policy doesn't control this!
        self._action_dim = 0  

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 0, device=self.device)

    @property
    def processed_actions(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 0, device=self.device)
        
    def process_actions(self, actions: torch.Tensor):
        pass # Fully autonomous; bypasses RL actions entirely
        
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self.pid_vel.reset(env_ids)
        self.pid_angle.reset(env_ids)
        self.pid_gyro.reset(env_ids)
        if env_ids is None:
            self.prev_lean.zero_()
        else:
            self.prev_lean[env_ids] = 0.0

    def _print_diagnostics(self, lean, lean_rate):
        """Prints the physical state of the first environment bike."""
        env_id = 0
        height = self.robot.data.root_pos_w[env_id, 2].item()
        lean_deg = torch.rad2deg(lean[env_id]).item()
        steer_pos = self.robot.data.joint_pos[env_id, self.steering_idx].item()
        fly_vel = self.robot.data.joint_vel[env_id, self.flywheel_idx].item()
        
        print("-" * 50)
        print(f"[BIKE DIAGNOSTICS - ENV {env_id}]")
        print(f"  Height (Z):    {height:7.4f} m  (Target ~0.42)")
        print(f"  Lean Angle:    {lean_deg:7.2f} deg")
        print(f"  Lean Rate:     {lean_rate[env_id]:7.2f} rad/s")
        print(f"  Steer Pos:     {steer_pos:7.4f} rad")
        print(f"  Flywheel Vel:  {fly_vel:7.2f} rad/s")
        
        # Safely extract the RL agent's commands from the global action manager
        rl_actions = self._env.action_manager.action
        rl_steer = rl_actions[env_id, 0].item() if rl_actions.shape[1] > 0 else 0.0
        rl_drive = rl_actions[env_id, 1].item() if rl_actions.shape[1] > 1 else 0.0
        
        print(f"  RL Steer Cmd:  {rl_steer:7.4f}")
        print(f"  RL Drive Cmd:  {rl_drive:7.4f}")
        
        # .item() appended safely format tensors to floats
        print(f"  Flywheel Torque: {self.robot.data.applied_torque[env_id, self.flywheel_idx].item():7.4f} Nm")
        print(f"  Steering Torque: {self.robot.data.applied_torque[env_id, self.steering_idx].item():7.4f} Nm")
        print(f"  Drive Torque:    {self.robot.data.applied_torque[env_id, self.drive_idx].item():7.4f} Nm")
        
        if height < 0.2: print("  >> WARNING: Bike is SINKING!")
        if height > 0.6: print("  >> WARNING: Bike is JUMPING!") 
            
    def apply_actions(self):
        # Control frequency matches the RL environment step frequency
        dt = self._env.step_dt
        
        # 1. State Extraction via PyTorch
        # Slicing indices 3:7 extracts [w,x,y,z] from the guaranteed PyTorch state tensor!
        root_quats = self.robot.data.root_state_w[:, 3:7] 
        
        up_vec = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)
        # Updated to the new, faster Warp math function to clear the deprecation warning
        body_up = math_utils.quat_apply_inverse(root_quats, up_vec)
        
        lean = torch.atan2(body_up[:, 1], body_up[:, 2])
        lean_rate = (lean - self.prev_lean) / dt
        self.prev_lean = lean.clone()

        wheel_speed = self.robot.data.joint_vel[:, self.joint_idx]

        # 2. 3-Loop Cascade Math (Executes for all 16,384 envs simultaneously)
        target_lean_angle = self.pid_vel.step(0.0, wheel_speed, 0.0, dt)
        target_lean_rate = self.pid_angle.step(target_lean_angle, lean, 0.0, dt)
        pid_out = self.pid_gyro.step(target_lean_rate, lean_rate, 0.0, dt)

        flywheel_cmd = -1.0 * pid_out

        # 3. Safely inject effort into IsaacLab's physics pipeline
        self.robot.set_joint_effort_target(
            flywheel_cmd.unsqueeze(-1), 
            joint_ids=self.joint_idx_tensor
        )
        
        # 4. Trigger Diagnostics (e.g., once every 50 simulation steps)
        if int(self._env.common_step_counter) % 10 == 0:
            self._print_diagnostics(lean, lean_rate)

# ======================================================================
# 4. DEPLOY-MATCHED STEER DELAY + SLEW (env-step queue, then 2 rad/s slew)
# ======================================================================
class VelocityBlendedSteerScaleJointPositionAction(JointPositionAction):
    """Steer position with a0→δ scale lerped from slow→fast over commanded |vx*|.

    Used by Paper-Hybrid so standstill gets Balance-style ±90° authority while
    riding keeps Paper ±10°. Blend (not a hard switch) so δ* is continuous as
    ``ramp_velocity_command`` moves vx*:

        alpha = clamp((|vx*| - v_lo) / (v_hi - v_lo), 0, 1)
        scale = (1 - alpha) * scale_slow + alpha * scale_fast
        δ* = a0 * scale
    """

    cfg: "VelocityBlendedSteerScaleJointPositionActionCfg"

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        cmd = self._env.command_manager.get_command(self.cfg.command_name)[:, 0].abs()
        v_lo = float(self.cfg.v_lo_ms)
        v_hi = float(self.cfg.v_hi_ms)
        denom = max(v_hi - v_lo, 1.0e-6)
        alpha = ((cmd - v_lo) / denom).clamp(0.0, 1.0)
        scale = (1.0 - alpha) * float(self.cfg.scale_slow) + alpha * float(self.cfg.scale_fast)
        # Broadcast per-env scale onto joint dim (steer is 1-DoF).
        self._processed_actions = self._raw_actions * scale.unsqueeze(-1) + self._offset
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )


class DelayedSlewJointPositionAction(JointPositionAction):
    """Steer position target with per-env delay (1–4 env steps) then rate slew.

    Delay is on the **commanded** δ (same place deploy slews δd), not Ideal PD.
    Implicit actuator PD is unchanged.
    """

    cfg: "DelayedSlewJointPositionActionCfg"

    def __init__(self, cfg: "DelayedSlewJointPositionActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        buffer_history = max(int(cfg.max_delay_steps), int(cfg.delay_buffer_capacity))
        self._delay_buffer = DelayBuffer(buffer_history, env.num_envs, device=env.device)
        self._slew_pos = torch.zeros(env.num_envs, self.action_dim, device=env.device)
        self._ALL_INDICES = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
        self._resample_delay(self._ALL_INDICES)

    def _current_joint_pos(self) -> torch.Tensor:
        pos = self._asset.data.joint_pos
        pos = pos.torch if hasattr(pos, "torch") else pos
        return pos[:, self._joint_ids]

    def _resample_delay(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self._ALL_INDICES
        n = int(env_ids.numel()) if torch.is_tensor(env_ids) else len(env_ids)
        if n == 0:
            return
        lags = torch.randint(
            low=int(self.cfg.min_delay_steps),
            high=int(self.cfg.max_delay_steps) + 1,
            size=(n,),
            dtype=torch.int,
            device=self.device,
        )
        self._delay_buffer.set_time_lag(lags, env_ids)

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        delayed = self._delay_buffer.compute(self._processed_actions)
        dt = float(self._env.step_dt)
        max_step = float(self.cfg.slew_rad_s) * dt
        delta = torch.clamp(delayed - self._slew_pos, min=-max_step, max=max_step)
        self._slew_pos = self._slew_pos + delta
        self._processed_actions = self._slew_pos

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = self._ALL_INDICES
        self._resample_delay(env_ids)
        self._delay_buffer.reset(env_ids)
        # Snap slew to measured steer so the first ticks are not a slam from 0.
        # Do not seed via compute(): DelayBuffer.append is all-batch and would
        # extra-lag envs that are still running.
        self._slew_pos[env_ids] = self._current_joint_pos()[env_ids]


class RcDriveJointVelocityAction(JointVelocityAction):
    """Rear-wheel velocity from ``cmd_vx / r``. Policy ``a1`` is ignored.

    Written for RC-owned drive; Delay-v0 does not wire this term (policy ``a1``
    is applied instead). Swap it in if ``onnx_use_policy_drive`` stays False.
    """

    cfg: "RcDriveJointVelocityActionCfg"

    def apply_actions(self):
        cmd = self._env.command_manager.get_command(self.cfg.command_name)[:, 0]
        omega = self.cfg.drive_joint_sign * cmd / self.cfg.wheel_radius_m
        clip = float(self.cfg.velocity_clip_rad_s)
        omega = torch.clamp(omega, -clip, clip)
        self._processed_actions = omega.unsqueeze(-1)
        self._asset.set_joint_velocity_target_index(target=self._processed_actions, joint_ids=self._joint_ids)


class SlewLimitedJointVelocityAction(JointVelocityAction):
    """Joint-velocity target with linear ground-speed acceleration slew.

    Policy still outputs a1 ∈ [-1, 1] → ω* = a1 · scale, but the applied target
    can change by at most ``accel_ms2 / wheel_radius_m`` rad/s each second
    (e.g. 1.0 m/s² → ~13.8 rad/s² on a 0.0725 m wheel). Cuts bang-bang hub
    commands without changing the action observation scale.
    """

    cfg: "SlewLimitedJointVelocityActionCfg"

    def __init__(self, cfg: "SlewLimitedJointVelocityActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._slew_vel = torch.zeros(env.num_envs, self.action_dim, device=env.device)
        self._ALL_INDICES = torch.arange(env.num_envs, dtype=torch.long, device=env.device)

    def _current_joint_vel(self) -> torch.Tensor:
        vel = self._asset.data.joint_vel
        vel = vel.torch if hasattr(vel, "torch") else vel
        return vel[:, self._joint_ids]

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)
        dt = float(self._env.step_dt)
        max_omega_dot = float(self.cfg.accel_ms2) / float(self.cfg.wheel_radius_m)
        max_step = max_omega_dot * dt
        delta = torch.clamp(self._processed_actions - self._slew_vel, min=-max_step, max=max_step)
        self._slew_vel = self._slew_vel + delta
        self._processed_actions = self._slew_vel

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = self._ALL_INDICES
        # Snap to measured hub speed so the first ticks are not a slam from 0.
        self._slew_vel[env_ids] = self._current_joint_vel()[env_ids]


class SlewLimitedCommandDriveAction(JointVelocityAction):
    """Hub ω* from velocity command (RC-style), with ground-speed accel slew.

    Matches deploy ``onnx_use_policy_drive=False``: speed is owned by v*, not by
    policy a1. Policy a1 is ignored for the plant (kept in the action vector for
    a 2-D ONNX contract). Applied target changes by at most ``accel_ms2`` m/s².
    """

    cfg: "SlewLimitedCommandDriveActionCfg"

    def __init__(self, cfg: "SlewLimitedCommandDriveActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._slew_vel = torch.zeros(env.num_envs, self.action_dim, device=env.device)
        self._ALL_INDICES = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
        n = env.num_envs
        rng = getattr(cfg, "accel_ms2_range", None)
        if rng is not None:
            self._accel_ms2 = torch.linspace(float(rng[0]), float(rng[1]), n, device=env.device)
        else:
            self._accel_ms2 = torch.full((n,), float(cfg.accel_ms2), device=env.device)

    def set_accel_ms2(self, env_ids: torch.Tensor, values: torch.Tensor) -> None:
        """Pin per-env ground-speed accel (m/s²) for drive-accel sweep eval."""
        self._accel_ms2[env_ids] = values.to(device=self._accel_ms2.device, dtype=self._accel_ms2.dtype)

    def _current_joint_vel(self) -> torch.Tensor:
        vel = self._asset.data.joint_vel
        vel = vel.torch if hasattr(vel, "torch") else vel
        return vel[:, self._joint_ids]

    def process_actions(self, actions: torch.Tensor):
        # Still call super so action buffers / logging stay consistent, then
        # overwrite the target from the velocity command.
        super().process_actions(actions)
        cmd = self._env.command_manager.get_command(self.cfg.command_name)[:, 0]
        r = float(self.cfg.wheel_radius_m)
        omega = float(self.cfg.drive_joint_sign) * cmd / r
        clip = float(self.cfg.velocity_clip_rad_s)
        omega = torch.clamp(omega, -clip, clip).unsqueeze(-1)

        dt = float(self._env.step_dt)
        max_step = (self._accel_ms2.unsqueeze(-1) / r) * dt
        delta = torch.clamp(omega - self._slew_vel, min=-max_step, max=max_step)
        self._slew_vel = self._slew_vel + delta
        self._processed_actions = self._slew_vel

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = self._ALL_INDICES
        self._slew_vel[env_ids] = self._current_joint_vel()[env_ids]


@configclass
class VelocityBlendedSteerScaleJointPositionActionCfg(JointPositionActionCfg):
    """Cfg for :class:`VelocityBlendedSteerScaleJointPositionAction`.

    ``scale`` on the base cfg is unused at runtime (blend owns a0→δ). Keep a
    sensible default for eval/IO metadata (typically ``scale_fast``).
    """

    class_type: type = VelocityBlendedSteerScaleJointPositionAction
    command_name: str = "velocity"
    scale_slow: float = 0.17453292519943295  # ±10° at train start; curriculum may raise
    scale_fast: float = 0.17453292519943295  # ±10°
    v_lo_ms: float = 0.0
    v_hi_ms: float = 0.5


@configclass
class DelayedSlewJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = DelayedSlewJointPositionAction
    min_delay_steps: int = 1
    max_delay_steps: int = 4
    delay_buffer_capacity: int = 4
    slew_rad_s: float = 2.0


@configclass
class RcDriveJointVelocityActionCfg(JointVelocityActionCfg):
    class_type: type = RcDriveJointVelocityAction
    command_name: str = "velocity"
    wheel_radius_m: float = 0.0725
    drive_joint_sign: float = 1.0
    velocity_clip_rad_s: float = 45.0


@configclass
class SlewLimitedJointVelocityActionCfg(JointVelocityActionCfg):
    class_type: type = SlewLimitedJointVelocityAction
    wheel_radius_m: float = 0.0725
    accel_ms2: float = 1.0  # max |dv/dt| of ground speed implied by ω*


@configclass
class SlewLimitedCommandDriveActionCfg(JointVelocityActionCfg):
    class_type: type = SlewLimitedCommandDriveAction
    command_name: str = "velocity"
    wheel_radius_m: float = 0.0725
    drive_joint_sign: float = 1.0
    velocity_clip_rad_s: float = 45.0
    accel_ms2: float = 1.0
    accel_ms2_range: tuple[float, float] | None = None  # per-env linspace if set


@configclass
class CascadeFlywheelActionCfg(ActionTermCfg):
    # Defined AFTER the Action class, so we can map it directly!
    class_type: type = CascadeFlywheelAction 
    asset_name: str = "robot"
    joint_name: str = "Revolute_11"
    
    # Outer Loop: Velocity (Momentum Dumping)
    vel_kp: float = 0.00001
    vel_ki: float = 0.0002
    vel_kd: float = 0.0
    vel_out_limit: float = 0.1
    vel_i_limit: float = 2000.0
    
    # Middle Loop: Angle (Balancing)
    angle_kp: float = 15.0
    angle_ki: float = 0.0
    angle_kd: float = 1.0
    angle_out_limit: float = 5.0
    angle_i_limit: float = 1.0
    
    # Inner Loop: Gyro (Torque output)
    gyro_kp: float = 30.0
    gyro_ki: float = 0.0
    gyro_kd: float = 0.5
    gyro_out_limit: float = 20.0
    gyro_i_limit: float = 10.0