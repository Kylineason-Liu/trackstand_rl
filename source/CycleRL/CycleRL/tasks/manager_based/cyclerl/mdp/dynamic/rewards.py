import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv

from ..rewards import action_rate_penalty

def survival_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)

def velocity_tracking_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    actual_vel = asset.data.root_lin_vel_b[:, 0] # Forward velocity (X)
    cmd_vel = env.command_manager.get_command("velocity")[:, 0]
    
    alpha = 0.25 # From paper
    error = torch.abs(actual_vel - cmd_vel)
    return torch.exp(-alpha * error)

def steering_tracking_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    steer_joint_name: str = "Revolute_7",
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    # Simplified: comparing steering joint position to command
    actual_steer = asset.data.joint_pos[:, asset.find_joints(steer_joint_name)[0][0]]

    # CRITICAL FIX: Index 2 is the steering/heading target! Index 0 would always be 0.0
    cmd_steer = env.command_manager.get_command("steering")[:, 2]

    beta = 0.1  # From paper
    error = torch.abs(actual_steer - cmd_steer)
    return torch.exp(-beta * error)

def action_magnitude_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    # Penalize the magnitude of the full action vector.
    return -torch.norm(env.action_manager.action, p=1, dim=-1)

# Flywheel Only
def upright_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward keeping the bicycle upright by analyzing the projected gravity vector."""
    asset = env.scene[asset_cfg.name]
    projected_gravity = asset.data.projected_gravity_b
    tilt = torch.abs(projected_gravity[:, 1])
    
    # Allow up to ~1.0 degree of tilt (sin(1.0 deg) ≈ 0.017) without penalty.
    tilt_with_deadzone = torch.clamp(tilt - 0.017, min=0.0)
    
    return torch.exp(-5.0 * tilt_with_deadzone)

def flywheel_saturation_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize high velocities on the flywheel joint to prevent infinite spinning."""
    asset = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(joint_vel), dim=-1)

def weighted_action_magnitude_penalty(
    env: ManagerBasedRLEnv, 
    weights: list[float]
) -> torch.Tensor:
    """Penalize action magnitude, but apply different multipliers to different motors."""
    w = torch.tensor(weights, device=env.device)
    # Multiply the actions by our custom weights before calculating the penalty
    return -torch.sum(w * torch.abs(env.action_manager.action), dim=-1)

def weighted_action_rate_penalty(
    env: ManagerBasedRLEnv, 
    weights: list[float]
) -> torch.Tensor:
    """Penalize rapid twitching, but apply different multipliers to different motors."""
    w = torch.tensor(weights, device=env.device)
    action_diff = env.action_manager.action - env.action_manager.prev_action
    # Multiply the twitching by our custom weights before calculating the penalty
    return -torch.norm(w * action_diff, p=2, dim=-1)


def no_jumping_penalty_local(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    Penalize vertical velocity relative to the bicycle's chassis (Body Frame).
    This perfectly allows driving up ramps/slopes, but heavily punishes bunny hopping!
    """
    asset = env.scene[asset_cfg.name]
    # Use '_b' (Body frame) instead of '_w' (World frame)
    # Index 2 is the local Z-axis (pointing straight up from the bike's perspective)
    local_z_vel = asset.data.root_lin_vel_b[:, 2]
    
    # Square it to exponentially punish large hops while ignoring tiny road vibrations
    return torch.square(local_z_vel)

def flywheel_oscillation_penalty(env: ManagerBasedRLEnv, flywheel_idx: int) -> torch.Tensor:
    """
    Specifically target the flywheel to act as a mathematical low-pass filter.
    Punishes violent frame-to-frame torque reversals exponentially.
    """
    # Extract only the action commands for the flywheel
    action = env.action_manager.action[:, flywheel_idx]
    prev_action = env.action_manager.prev_action[:, flywheel_idx]
    
    # Square the difference to create the exponential "high-frequency" wall
    return torch.square(action - prev_action)

def flywheel_trackstand_bonus(
    env: ManagerBasedRLEnv, 
    flywheel_idx: int, 
    command_name: str, 
    vel_threshold: float
) -> torch.Tensor:
    """
    Rewards the agent slightly for utilizing the flywheel when the target velocity is near zero.
    Acts as a 'breadcrumb' to teach the agent to rely on the reaction wheel for trackstands.
    """
    # 1. Get the current forward (X-axis) velocity command
    # (Update "command_name" in the config if your command term is named differently)
    velocity_command = env.command_manager.get_command(command_name)[:, 0]
    
    # 2. Check if we are currently in a trackstand phase
    is_trackstand = torch.abs(velocity_command) <= vel_threshold
    
    # 3. Get the absolute magnitude of the flywheel action
    flywheel_action_mag = torch.abs(env.action_manager.action[:, flywheel_idx])
    
    # 4. Pay out the reward ONLY if the trackstand condition is true
    return flywheel_action_mag * is_trackstand.float()

def trackstand_stationary_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "velocity",
    vel_threshold: float = 0.01,
) -> torch.Tensor:
    """
    When the velocity command is near zero (trackstand), penalize the bike
    for drifting away from its spawn position. This encourages the bike to
    maintain its ground position while balancing at a standstill.

    Penalty scales quadratically with displacement — small wobbles are
    tolerated, but large drifts are heavily discouraged.

    Only active when |velocity_command| < vel_threshold (default 0.01 m/s).
    """
    asset = env.scene[asset_cfg.name]

    # Current XY position in world frame
    current_pos = asset.data.root_pos_w[:, :2]

    # Initial spawn position (env origins, since pose_range x/y are always 0)
    initial_pos = env.scene.env_origins[:, :2]

    # Squared displacement from spawn — quadratic penalty for large drifts
    displacement_sq = torch.sum(torch.square(current_pos - initial_pos), dim=-1)

    # Only apply penalty during trackstand (velocity command ≈ zero)
    velocity_command = env.command_manager.get_command(command_name)[:, 0]
    is_trackstand = torch.abs(velocity_command) < vel_threshold

    return displacement_sq * is_trackstand.float()


def strict_trackstand_motor_penalty(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    vel_threshold: float,
    motor_indices: list[int]
) -> torch.Tensor:
    """
    Heavily penalizes using the steering and drive motors when the target velocity is near zero.
    This traps the agent into relying solely on the flywheel for trackstand balancing.
    """
    # 1. Get the current forward (X-axis) velocity command
    velocity_command = env.command_manager.get_command(command_name)[:, 0]
    
    # 2. Create a boolean mask: True if we are supposed to be standing still
    is_trackstand = torch.abs(velocity_command) < vel_threshold
    
    # 3. Extract the action commands specifically for the forbidden motors
    forbidden_actions = env.action_manager.action[:, motor_indices]
    
    # 4. Calculate the magnitude of these actions (squaring them punishes large twitches)
    action_penalty = torch.sum(torch.square(forbidden_actions), dim=-1)
    
    # 5. Multiply the penalty by the mask so it ONLY applies during a trackstand
    return action_penalty * is_trackstand.float()