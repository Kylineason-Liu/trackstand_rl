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


def balance_roll_shaping(
    env: ManagerBasedRLEnv,
    sigma: float = 0.1396,  # 8 deg: trackstand-quality uprightness kernel (survey w2*exp(-phi^2/2s^2))
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian uprightness shaping: exp(-roll^2/(2 sigma^2)) in (0, 1].

    1.0 at perfectly upright, decaying smoothly with lean. Unlike the binary
    survival term this gives a *gradient toward uprightness* inside the survival
    band (survival pays 29 deg the same as 0 deg; this term does not). The kernel
    is the survey's prescribed uprightness term (Baltes' 1-phi^2 and Weyrer's
    (eta-|phi|)/eta are equivalent alternatives).
    """
    asset = env.scene[asset_cfg.name]
    roll = torch.asin(asset.data.projected_gravity_b[:, 1].clamp(-1.0, 1.0))
    return torch.exp(-(roll**2) / (2.0 * sigma**2))


def balance_displacement_band(
    env: ManagerBasedRLEnv,
    band: float = 0.2,   # full reward inside +-0.2 m of spawn (survey r_displacement band)
    sigma: float = 0.3,  # Gaussian decay beyond the band
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position-keeping as a TOLERANCE BAND, not a hard quadratic.

    1.0 while the bike stays within ``band`` of its spawn; decays as a Gaussian
    beyond. Track stand REQUIRES micro-motion (Brockett: no static feedback law
    exists), so a hard position penalty would fight the necessary oscillation —
    the band rewards staying put *within* a neighbourhood and only punishes
    genuine drift (the survey's "reward micro-motion, penalise drift" rule).
    """
    asset = env.scene[asset_cfg.name]
    d = torch.norm(asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=-1)
    excess = torch.clamp(d - band, min=0.0)
    return torch.exp(-(excess**2) / (2.0 * sigma**2))


def balance_yaw_band(
    env: ManagerBasedRLEnv,
    band: float = 0.1,  # rad (~6 deg): free reorientation allowance
    tail_slope: float = 0.3288,  # = 1/(pi - band): value hits 0 at |yaw| = pi (full reversal)
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Plateau + LINEAR-TAIL band on |yaw| relative to the spawn heading.

    ANTI-ORBITAL term. A track stand oscillates around a fixed heading, but a
    policy holding a constant steer deflection + roll rides a CIRCLE around the
    spawn: radial drift metrics (and drift_band) are blind to orbital motion,
    so a circle inside the drift band earns full position credit forever and
    only the horizon ends the episode.

    Iter3 used a Gaussian (sigma=0.4 rad) here and FAILED: once |yaw| passed
    ~0.8 rad the Gaussian hit ~0 and its gradient vanished, so the policy
    forfeited the whole term and orbited freely (TB yaw_band = 1.7e-12;
    eval turn/episode = 282 deg).

    Iter4 used a linear tail CLAMPED at 0, and the policy parked EXACTLY at the
    clamp floor: with slope 1/(pi-band) the value hits 0 at |yaw| = pi, and
    eval yaw sat at 174.7 deg ~ pi with zero gradient. The value is therefore
    NOT clamped at all now: it goes NEGATIVE beyond the band. Since |yaw| <= pi
    from atan2, the term is bounded in [-1, 1] — no overflow, no suicide risk,
    and a CONSTANT gradient everywhere: no parking spot exists.

    BASELINE (iter6 audit finding): root_quat_w is stored (x, y, z, w) in this
    stack (isaaclab's own quat_from_euler_xyz returns [qx,qy,qz,qw]; physx
    passthrough is xyzw), while the earlier hand-rolled formula assumed
    (w, x, y, z) — on the identity spawn quaternion it returned pi, which
    misread "upright at spawn" as "facing backward". The canonical
    ``asset.data.heading_w`` (base-frame +x rotated to world, atan2(y, x) —
    kernels.py root_heading_w) replaces the hand-rolled math. The env
    (NewBikeBalanceHybridEnv) records ``spawn_yaw`` per env at reset; this
    term measures the wrap-aware yaw RELATIVE to that baseline.
    """
    asset = env.scene[asset_cfg.name]
    yaw = asset.data.heading_w  # canonical: atan2 of base +x in world (rad)
    rel = torch.atan2(torch.sin(yaw - env.spawn_yaw), torch.cos(yaw - env.spawn_yaw))
    excess = torch.clamp(rel.abs() - band, min=0.0)
    return 1.0 - tail_slope * excess


def balance_wheel_net_angle_band(
    env: ManagerBasedRLEnv,
    band: float = 1.0,  # rad of free NET wheel rotation since reset
    tail_slope: float = 0.19,  # per rad beyond band
    penalty_floor: float = -1.0,  # cap the per-step penalty (bounded cost, no suicide pressure)
) -> torch.Tensor:
    """Plateau + linear tail on |NET drive-wheel angle| since reset, floor-capped.

    Zero-mean micro-motion requirement (the user's anti-orbit idea): oscillation
    keeps the net angle ~0, while constant rolling, orbiting, or sustained
    in-place wheel spin accumulate it monotonically and are pushed back with a
    constant gradient. Complements micro_band (instantaneous |w| cap) and
    yaw_band (heading) — this is the wheel-space backstop.

    NOTE: ``env.wheel_net_angle`` is maintained by the custom
    ``NewBikeBalanceHybridEnv`` (new_bike/newbike_balance_env.py), because the
    wheel's joint_pos WRAPS at +-pi in this stack (verified empirically:
    write 10.0 rad -> read back -2.5 rad) and a stateless reward function
    cannot integrate velocity itself. The env integrates joint_vel each step
    and zeroes the buffer for reset envs.

    The floor cap matters: an unbounded growing penalty resets to 0 on episode
    end, which teaches "fall deliberately to escape the penalty" (suicide).
    Capped at -1/step, surviving is always preferable to falling.
    """
    theta = env.wheel_net_angle  # (num_envs,), maintained by NewBikeBalanceHybridEnv
    excess = torch.clamp(theta.abs() - band, min=0.0)
    return (1.0 - tail_slope * excess).clamp(min=penalty_floor)


def balance_micro_motion_band(
    env: ManagerBasedRLEnv,
    target: float = 3.0,  # rad/s: free micro-motion allowance (Wang 2024 / 黄用华 2012)
    tail_slope: float = 0.03,  # per rad/s beyond target -> 0 at |w| = target + 33
    drive_joint_name: str = "rearwheel_joint",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Band reward on |rear wheel speed|: full credit up to ``target``, linear tail beyond.

    Wang 2024: track stand works through torque-level high-frequency rear-wheel
    micro-motion. Iteration 1 used a symmetric Gaussian (sigma=3.0), whose
    gradient vanishes beyond ~10 rad/s — the policy spun the wheel (eval: 15
    rad/s mean, 239 max, diverged deaths) and the term said nothing. The linear
    tail keeps a CONSTANT gradient at any speed, so runaway spin is always
    pushed back toward the micro-motion band, while up to ``target`` the wheel
    is free to oscillate (frozen wheel gets full credit too — the drift_band
    term handles the position side, per the survey's separation of concerns).
    """
    asset = env.scene[asset_cfg.name]
    idx = asset.find_joints(drive_joint_name)[0][0]
    w = asset.data.joint_vel[:, idx].abs()
    excess = torch.clamp(w - target, min=0.0)
    return (1.0 - tail_slope * excess).clamp(min=0.0)


def balance_action_rate_penalty_norm(
    env: ManagerBasedRLEnv,
    action_scales: tuple[float, ...] = (1.57, 6.28),  # steer rad, drive N-m
) -> torch.Tensor:
    """Per-joint-NORMALISED action-rate penalty: -sqrt(sum_j (da_j/scale_j)^2).

    The stock ``action_rate_penalty`` computes the plain L2 norm of the action
    change, which for the hybrid task mixes units (steering rad vs drive N-m) and
    weights them arbitrarily. Dividing each dimension by its action scale makes
    the penalty unit-free before summing. Negative and small; keep the weight
    low so the necessarily-oscillatory track stand control is not over-damped
    (survey: penalise amplitude excess, not frequency).
    """
    scales = torch.tensor(action_scales, device=env.device)
    da = env.action_manager.action - env.action_manager.prev_action
    return -torch.sqrt(((da / scales) ** 2).sum(dim=-1) + 1e-8)


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
