import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def thubike_fallen_termination(
    env: ManagerBasedRLEnv,
    limit_angle: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the bike falls over or flips.

    Two checks:
    1. Roll tilt — |gravity_y| > sin(limit_angle) means the bike leaned too far sideways.
    2. Inversion  — gravity_z > 0 means the bike is upside-down (pitch or roll flip past 90°).
    """
    asset = env.scene[asset_cfg.name]
    gravity_y = asset.data.projected_gravity_b[:, 1]
    gravity_z = asset.data.projected_gravity_b[:, 2]

    limit = torch.sin(torch.tensor(limit_angle, device=env.device))
    too_tilted = torch.abs(gravity_y) > limit
    flipped = gravity_z > 0.0  # bike is inverted (more than 90° from upright)

    return too_tilted | flipped


def balance_yaw_deviation_termination(
    env: ManagerBasedRLEnv,
    limit_angle: float = 1.0472,  # 60 deg, in rad (heading_w is radians)
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when |heading - spawn heading| exceeds ``limit_angle`` (rad).

    Heading is the canonical ``asset.data.heading_w`` (base +x rotated to
    world, atan2 — radians). The spawn baseline ``env.spawn_yaw`` is maintained
    per env by NewBikeBalanceHybridEnv, so this term only applies to tasks
    running on that env class.
    """
    asset = env.scene[asset_cfg.name]
    # raw root quat (xyzw, fresh in all windows) -> euler yaw; kernel-cached
    # heading_w is stale in the post-reset window (audit finding)
    yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w.torch)[2]
    rel = torch.atan2(torch.sin(yaw - env.spawn_yaw), torch.cos(yaw - env.spawn_yaw))
    return rel.abs() > limit_angle


def balance_wheel_net_termination(
    env: ManagerBasedRLEnv,
    limit_angle: float = 7.2,  # rad of net rear-wheel rotation since reset
) -> torch.Tensor:
    """Terminate when |net drive-wheel rotation| since reset exceeds the limit.

    Uses ``env.wheel_net_angle`` (rad) maintained by NewBikeBalanceHybridEnv —
    the raw joint_pos WRAPS at +-pi and cannot be used (verified empirically).
    One-step lag: the env integrates after super().step(), so this term reads
    the value through the PREVIOUS step — <= 0.03 rad at the 1.5 rad/s micro
    band; at runaway speeds the ``diverged`` term already covers that regime.
    """
    return env.wheel_net_angle.abs() > limit_angle


def balance_drift_termination(
    env: ManagerBasedRLEnv,
    max_dist: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reset a bike that drifts far from spawn (Weyrer's displacement-threshold idea).

    Track stand tolerates micro-motion but not genuine drift: without this, a
    policy can learn "keep rolling slowly sideways/forward" and never fall while
    the episode runs out the clock. 2 m is deliberately loose — well inside it,
    the displacement BAND reward shapes behaviour; only clear drift is reset.
    """
    asset = env.scene[asset_cfg.name]
    d = torch.norm(asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=-1)
    return d > max_dist


def nonfinite_or_extreme_state(
    env: ManagerBasedRLEnv,
    max_lin_vel: float = 20.0,
    max_ang_vel: float = 50.0,
    max_joint_vel: float = 300.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reset a diverging bike *before* it can NaN-poison the shared policy.

    A single env whose root state goes non-finite contaminates the batched
    observation, so the PPO update writes NaN into the shared network weights and
    the whole run dies. Firing this termination resets the bad env each step, so
    the blow-up is contained to one env instead of ending training.

    Checks the ROOT (lin/ang velocity) and the JOINTS. The joint check matters
    under torque control, where nothing else bounds a free-spinning wheel: the
    torque runs died with |drive_vel| = 2420 rad/s in the observation while the
    root state stayed perfectly finite, so a root-only check never fired.
    ``max_joint_vel`` is deliberately loose (5x the drive motor's 61.6 rad/s limit):
    PhysX velocity limits are soft constraints, so sustained full torque still
    overshoots to ~130 rad/s, and this net is meant to catch genuine divergence
    (10^3+), not ordinary saturation. Protecting the network from large-but-finite
    values is the observation clamp's job (see stationary/observations.py).
    """
    asset = env.scene[asset_cfg.name]
    lin_vel = asset.data.root_lin_vel_b
    ang_vel = asset.data.root_ang_vel_b
    joint_vel = asset.data.joint_vel
    nonfinite = (
        ~torch.isfinite(lin_vel).all(dim=-1)
        | ~torch.isfinite(ang_vel).all(dim=-1)
        | ~torch.isfinite(joint_vel).all(dim=-1)
    )
    too_fast = torch.norm(lin_vel, dim=-1) > max_lin_vel
    spinning = torch.norm(ang_vel, dim=-1) > max_ang_vel
    joint_runaway = joint_vel.abs().amax(dim=-1) > max_joint_vel
    return nonfinite | too_fast | spinning | joint_runaway
