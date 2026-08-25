import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from ..rewards import action_rate_penalty


def balance_upright_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exponential reward that decays sharply with roll tilt."""
    asset = env.scene[asset_cfg.name]
    roll_tilt = torch.abs(asset.data.projected_gravity_b[:, 1])
    return torch.exp(-10.0 * roll_tilt)


def balance_stability_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise angular velocity — encourages a smooth, still balance."""
    asset = env.scene[asset_cfg.name]
    ang_vel_mag = torch.norm(asset.data.root_ang_vel_b, dim=-1)
    return torch.exp(-2.0 * ang_vel_mag)


def balance_drift_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise linear drift from the origin so the bike stays in place."""
    asset = env.scene[asset_cfg.name]
    drift = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=-1)
    return -drift


def balance_action_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Small penalty to discourage thrashing the actuators."""
    return -torch.norm(env.action_manager.action, p=2, dim=-1)


def balance_action_rate_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalise high-frequency action changes for smooth control."""
    return action_rate_penalty(env)


def balance_joint_acc_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Light penalty on joint accelerations to discourage motor twitching."""
    asset = env.scene[asset_cfg.name]
    return -torch.norm(asset.data.joint_acc, p=2, dim=-1)


def balance_joint_torque_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Light penalty on raw motor effort for energy-efficient control."""
    return -torch.norm(env.action_manager.action, p=2, dim=-1)


def balance_position_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalise XY displacement from the spawn origin.

    Quadratic penalty — small wobbles are tolerated, large drifts are heavily
    discouraged. Always active so the bike prefers staying near its start.
    """
    asset = env.scene[asset_cfg.name]
    current_pos = asset.data.root_pos_w[:, :2]
    initial_pos = env.scene.env_origins[:, :2]
    displacement_sq = torch.sum(torch.square(current_pos - initial_pos), dim=-1)
    return -displacement_sq


def trackstand_position_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exponential penalty on XY displacement from the spawn origin.

    Near-zero near spawn (free to use small motions for balancing), but grows
    explosively beyond ~1 m — forcing the policy to learn backward motion and
    tight turns to return, rather than creeping forward indefinitely.
    """
    asset = env.scene[asset_cfg.name]
    current_pos = asset.data.root_pos_w[:, :2]
    initial_pos = env.scene.env_origins[:, :2]
    displacement = torch.norm(current_pos - initial_pos, dim=-1)
    return -(torch.exp(3.0 * displacement) - 1.0)


def balance_survival_reward(
    env: ManagerBasedRLEnv,
    decay_per_deg: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exponential uprightness reward in (0, 1]: ``exp(-decay_per_deg * |roll_deg|)``.

    +1.0 at vertical, decaying smoothly toward 0 as the bike leans (``exp(-3)≈0.05``
    at 30° with the default 0.1). Always positive and self-peaking at 1.0, so the cfg
    ``weight`` IS the peak reward (weight 5 => peak 5). Roll is converted to degrees
    first — the raw ``projected_gravity_b[:, 1] ≈ sin(roll)`` is only a lean fraction,
    so ``exp(-0.1*sin_lean)`` would be nearly flat. Steepest gradient is AT vertical
    and fades as the bike leans: precision near upright, weaker signal out in the fall
    zone (the opposite trade from the arctan variant kept below). No 30° gate needed —
    ``fallen`` terminates at 30° and :func:`balance_termination_penalty` supplies the
    failure cost. ``decay_per_deg`` is the tuning knob (smaller = gentler / more
    forgiving decay).
    """
    asset = env.scene[asset_cfg.name]
    roll_lean = torch.abs(asset.data.projected_gravity_b[:, 1]).clamp(max=1.0)  # ≈ |sin(roll)|
    roll_deg = torch.rad2deg(torch.asin(roll_lean))
    return torch.exp(-decay_per_deg * roll_deg)
    # arctan alternative (kept — needs `import math`): self-normalized S-curve, +1 at
    # vertical, 0 at midpoint_deg, negative past it; strongest gradient at the midpoint
    # (near the fall edge) rather than at vertical:
    # midpoint_deg, sharpness = 20.0, 0.15
    # norm = math.atan(sharpness * midpoint_deg)
    # return torch.atan(sharpness * (midpoint_deg - roll_deg)) / norm


def balance_survival_reward_binary(
    env: ManagerBasedRLEnv,
    limit_deg: float = 30.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Binary alive bonus: 1.0 while ``|roll| < limit_deg``, else 0.0.

    The simplest survival reward — a flat +1 per step for staying upright (within
    30°), with NO shaping. Peaks at 1.0, so the cfg ``weight`` is the per-step alive
    bonus. Being a step function it has no gradient w.r.t. roll, so it teaches "don't
    fall" purely through accumulated alive-time (paired with the termination penalty)
    and gives no per-step "more upright is better" signal — the exp/arctan variants
    above do.
    """
    asset = env.scene[asset_cfg.name]
    roll_lean = torch.abs(asset.data.projected_gravity_b[:, 1]).clamp(max=1.0)  # ≈ |sin(roll)|
    roll_deg = torch.rad2deg(torch.asin(roll_lean))
    return (roll_deg < limit_deg).float()


def balance_termination_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant -1 on the step a *failure* termination fires, else 0.

    Reads ``env.termination_manager.terminated`` — the non-time_out terminations
    ('fallen', 'diverged') — so surviving to the episode time-out is NOT
    penalised. A one-time terminal cost (unlike a per-step fall penalty).
    """
    return -env.termination_manager.terminated.float()


def balance_drive_reward(
    env: ManagerBasedRLEnv,
    drive_joint_name: str = "rearwheel_joint",
    decay: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for keeping the rear (drive) wheel calm; the cfg ``weight`` scales it.

    ``exp(-decay * |v|)`` in (0, 1]: 1.0 when the wheel is still, decaying smoothly
    as it spins (``exp(-2.5)≈0.08`` at 10 rad/s with the default 0.25). Bounded and
    always positive (a negative exponent can't overflow), so it can't explode or
    dominate the survival reward (peak 5) — it just gently rewards a slow drive
    wheel. Peaks at 1.0, so set ``weight`` to the desired peak reward.
    """
    asset = env.scene[asset_cfg.name]
    idx = asset.find_joints(drive_joint_name)[0][0]
    v = asset.data.joint_vel[:, idx]
    return torch.exp(-decay * torch.abs(v))


def balance_steer_reward(
    env: ManagerBasedRLEnv,
    steer_joint_name: str = "steering_joint",
    decay: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for keeping the steering motor calm; the cfg ``weight`` scales it.

    ``exp(-decay * |v|)`` in (0, 1]: 1.0 when steering is still, decaying as it
    moves. Bounded and always positive (a negative exponent can't overflow), so it
    can't explode or dominate survival (peak 5). Keep the weight small — steering
    is the bike's main balancing actuator, so an over-strong "stay still" reward
    suppresses the corrective steering it needs to stay upright. Peaks at 1.0.
    """
    asset = env.scene[asset_cfg.name]
    idx = asset.find_joints(steer_joint_name)[0][0]
    v = asset.data.joint_vel[:, idx]
    return torch.exp(-decay * torch.abs(v))


def balance_velocity_tracking_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for tracking the commanded velocity (forward or backward).

    Uses a wide exponential so small errors (~0.1 m/s) are still well-rewarded
    — the policy isn't punished for imperfect tracking while learning to balance.
    """
    asset = env.scene[asset_cfg.name]
    actual_vel = asset.data.root_lin_vel_b[:, 0]
    cmd_vel = env.command_manager.get_command("velocity")[:, 0]
    error = torch.abs(actual_vel - cmd_vel)
    return torch.exp(-0.5 * error)
