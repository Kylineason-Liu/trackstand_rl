import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def imu_orientation_and_velocity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    steer_joint_name: str = "Revolute_7",
) -> torch.Tensor:
    """Extracts a 6-D IMU block: gravity_y, wx, wy, wz, forward speed, wrapped steer."""
    asset = env.scene[asset_cfg.name]

    # Use projected gravity in the body frame to represent lean orientation.
    gravity_y = asset.data.projected_gravity_b[:, 1:2]

    # Root angular velocity in the local frame: roll, pitch, yaw.
    ang_vel = asset.data.root_ang_vel_b

    # Forward velocity in the local frame.
    forward_vel = asset.data.root_lin_vel_b[:, 0:1]

    # Steering joint position, wrapped to [-π, π] for continuous 360° forks.
    steering_joint_idx = asset.find_joints(steer_joint_name)[0][0]
    steering_pos = asset.data.joint_pos[:, steering_joint_idx:steering_joint_idx + 1]
    steering_pos = torch.atan2(torch.sin(steering_pos), torch.cos(steering_pos))

    return torch.cat([gravity_y, ang_vel, forward_vel, steering_pos], dim=-1)


def command_tracking_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Extracts the difference between commanded velocity/steering and actual."""
    vel_cmd = env.command_manager.get_command("velocity")
    steer_cmd = env.command_manager.get_command("steering")

    # vel_cmd[:, 0] is the commanded forward velocity (lin_vel_x)
    # steer_cmd[:, 2] is the commanded heading error (if heading_command=True)
    target_vel_x = vel_cmd[:, 0].unsqueeze(-1)
    target_heading = steer_cmd[:, 2].unsqueeze(-1)

    # Returns a dense tensor of shape (num_envs, 2)
    return torch.cat([target_vel_x, target_heading], dim=-1)
