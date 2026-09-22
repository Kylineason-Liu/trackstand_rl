from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

def fade_trackstand_ratio(
    env: ManagerBasedRLEnv, 
    env_ids: Sequence[int], 
    step_thresholds: list[int], 
    ratios: list[float]
) -> None:
    """
    Dynamically adjusts the rel_standing_envs ratio of the command generators
    based on the number of control steps taken by the environment.
    
    Args:
        step_thresholds: List of env.step() counts at which to change the ratio.
        ratios: The corresponding rel_standing_envs ratios to apply.
    """
    # env.common_step_count tracks how many times env.step() has been called
    current_step = int(env.common_step_counter)
    
    # Determine the target ratio for the current step
    target_ratio = ratios[0]
    for threshold, ratio in zip(step_thresholds, ratios):
        if current_step >= threshold:
            target_ratio = ratio
            
    # Apply the new ratio to BOTH the velocity and steering command generators
    for cmd_name in ["velocity", "steering"]:
        if cmd_name in env.command_manager._terms:
            cmd_term = env.command_manager._terms[cmd_name]
            
            # Update the configuration if it has changed
            if cmd_term.cfg.rel_standing_envs != target_ratio:
                cmd_term.cfg.rel_standing_envs = target_ratio
                
                # Optional: Print to terminal so you know the curriculum advanced!
                if current_step % 100 == 0:  # Prevent console spam
                    print(f"\n[Curriculum] Step {current_step}: Trackstand ratio reduced to {target_ratio * 100}%\n")

def expand_curriculum_range(
    env: ManagerBasedRLEnv, 
    env_ids: Sequence[int], 
    step_thresholds: list[int], 
    max_values: list[float],
    target_param: str  # The string flag to tell it what to update!
) -> None:
    """
    A unified curriculum function that dynamically expands various ranges 
    (spawn velocities, spawn headings, or command targets) based on training steps.
    """
    current_step = int(env.common_step_counter)
    
    # 1. Calculate the target value for the current step
    target_val = max_values[0]
    for threshold, val in zip(step_thresholds, max_values):
        if current_step >= threshold:
            target_val = val
            
    # 2. Apply the update based on the specified target_param flag
    
    # CASE A: Spawn Velocity — expand the upper bound, keep the existing minimum
    # (do not force 0 m/s; that is a trackstand spawn).
    if target_param == "spawn_velocity_x":
        if hasattr(env.event_manager.cfg, "initial_state"):
            event_params = env.event_manager.cfg.initial_state.params
            lo, hi = event_params["velocity_range"]["x"]
            if hi != target_val:
                event_params["velocity_range"]["x"] = (lo, target_val)
                if current_step % 100 == 0:
                    print(
                        f"\n[Curriculum] Step {current_step}: Spawn velocity X expanded to "
                        f"({lo}, {target_val}) m/s"
                    )

    # CASE B: Spawn Heading (Symmetric: -X to X)
    elif target_param == "spawn_yaw":
        if hasattr(env.event_manager.cfg, "initial_state"):
            event_params = env.event_manager.cfg.initial_state.params
            if event_params["pose_range"]["yaw"][1] != target_val:
                event_params["pose_range"]["yaw"] = (-target_val, target_val)
                if current_step % 100 == 0:
                    print(f"\n[Curriculum] Step {current_step}: Spawn yaw expanded to ({-target_val}, {target_val}) rad")

    # CASE C: Command Heading (Symmetric: -X to X)
    elif target_param == "command_heading":
        if "steering" in env.command_manager._terms:
            cmd_cfg = env.command_manager._terms["steering"].cfg
            if cmd_cfg.ranges.heading[1] != target_val:
                cmd_cfg.ranges.heading = (-target_val, target_val)
                if current_step % 100 == 0:
                    print(f"\n[Curriculum] Step {current_step}: Command heading expanded to ({-target_val}, {target_val}) rad")

    # CASE D: IMU observation Gaussian std (policy imu_state corruption)
    elif target_param == "imu_noise_std":
        if _set_imu_noise_std(env, target_val) and current_step % 100 == 0:
            print(f"\n[Curriculum] Step {current_step}: IMU noise std expanded to {target_val}\n")


def _iter_obs_groups(observations):
    """Yield observation groups from a dict or a configclass."""
    if observations is None:
        return
    if isinstance(observations, dict):
        yield from observations.values()
        return
    group = getattr(observations, "policy", observations)
    if group is not None:
        yield group


def _set_imu_noise_std(env: ManagerBasedRLEnv, std: float) -> bool:
    """Write Gaussian ``std`` onto live ``imu_state`` terms. Returns True if it changed."""
    changed = False
    obs_man = getattr(env, "observation_manager", None)
    if obs_man is not None:
        names_by_group = getattr(obs_man, "_group_obs_term_names", {})
        cfgs_by_group = getattr(obs_man, "_group_obs_term_cfgs", {})
        for group_name, names in names_by_group.items():
            if "imu_state" not in names:
                continue
            term_cfg = cfgs_by_group[group_name][names.index("imu_state")]
            noise = getattr(term_cfg, "noise", None)
            if noise is not None and hasattr(noise, "std") and noise.std != std:
                noise.std = std
                changed = True
    for group in _iter_obs_groups(getattr(env.cfg, "observations", None)):
        imu = getattr(group, "imu_state", None)
        noise = getattr(imu, "noise", None) if imu is not None else None
        if noise is not None and hasattr(noise, "std") and noise.std != std:
            noise.std = std
            changed = True
    return changed


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "velocity",
) -> torch.Tensor:
    """Move robots to harder / easier terrain rows from distance traveled.

    Progresses a robot if it reached half the sub-terrain width from spawn.
    Demotes it if it covered less than half the distance the velocity command
    required over the episode. Same rule as Isaac Lab locomotion, but this
    env's command term is ``velocity`` rather than ``base_velocity``.
    """
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    command = env.command_manager.get_command(command_name)
    distance = torch.linalg.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = distance < torch.linalg.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())

