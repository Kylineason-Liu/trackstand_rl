from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers import ActionTermCfg
from isaaclab.envs import ManagerBasedEnv
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
# 3. CUSTOM ACTION TERM CONFIGURATION
# ======================================================================
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