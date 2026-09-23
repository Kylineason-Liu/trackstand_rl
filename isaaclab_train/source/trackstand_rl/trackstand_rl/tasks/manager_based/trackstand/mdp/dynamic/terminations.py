import math
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def bicycle_lean_termination(
    env: ManagerBasedRLEnv,
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Terminates when the bicycle leans too far sideways.
    Because the URDF is Y-Up and rotated 90 degrees to fit Isaac Sim,
    we check the Y-axis (index 1) of the projected gravity vector.
    """
    asset = env.scene[asset_cfg.name]

    # projected_gravity_b[:, 1] is -1.0 when perfectly upright.
    # As the bike falls over, this value moves towards 0.0.
    return asset.data.projected_gravity_b[:, 2] > -math.cos(limit_angle)


def roll_exceeds_threshold(env: ManagerBasedRLEnv, threshold: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate the episode if the base roll angle exceeds the threshold (in degrees)."""
    asset = env.scene[asset_cfg.name]
    projected_gravity = asset.data.projected_gravity_b
    limit = math.sin(math.radians(threshold))
    return torch.abs(projected_gravity[:, 1]) > limit


def velocity_goal_timeout(
    env: ManagerBasedRLEnv,
    timeout_s: float = 5.0,
    error_threshold: float = 0.5,
    command_name: str = "velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate if forward speed stays below the velocity command for ``timeout_s``.

    The underspeed clock resets whenever the bike reaches the goal
    (``v >= v* - error_threshold``) and on env reset.
    """
    asset = env.scene[asset_cfg.name]
    cmd_vel = env.command_manager.get_command(command_name)[:, 0]
    actual_vel = asset.data.root_lin_vel_b[:, 0]
    at_goal = actual_vel >= (cmd_vel - error_threshold)

    n = env.num_envs
    device = env.device
    fail_time = getattr(env, "_vel_goal_fail_time", None)
    if fail_time is None or fail_time.shape[0] != n:
        fail_time = torch.zeros(n, device=device)

    dt = float(env.step_dt)
    just_reset = env.episode_length_buf <= 1
    fail_time = torch.where(just_reset | at_goal, torch.zeros_like(fail_time), fail_time + dt)
    env._vel_goal_fail_time = fail_time
    return fail_time >= timeout_s
