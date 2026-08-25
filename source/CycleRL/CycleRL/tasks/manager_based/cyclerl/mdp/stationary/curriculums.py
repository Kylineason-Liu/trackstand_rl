"""Diagnostic curriculum terms for the stationary balance task.

These change nothing about training — they print quantities to the terminal.
(skrl does NOT forward the env's ``extras["log"]`` to TensorBoard, so a value
returned from a curriculum term would not surface there — hence the prints.)
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv


def log_motor_speeds(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor | None, print_interval: int = 1000
) -> None:
    """Print per-episode mean/max drive & steer motor speed (rad/s) to the terminal.

    Reads the accumulators filled every step by
    ``mdp.stationary.observations._accumulate_motor_speeds`` for the envs that
    just finished (``env_ids``), zeros them, and prints roughly every
    ``print_interval`` env-steps. Prints (not TensorBoard) because skrl doesn't
    forward the env's ``extras["log"]``.
    """
    if not hasattr(env, "_motor_speed_count") or env_ids is None or len(env_ids) == 0:
        return
    cnt = env._motor_speed_count[env_ids].clamp(min=1.0)
    drive_mean = (env._drive_speed_sum[env_ids] / cnt).mean().item()
    drive_max = env._drive_speed_max[env_ids].max().item()
    steer_mean = (env._steer_speed_sum[env_ids] / cnt).mean().item()
    steer_max = env._steer_speed_max[env_ids].max().item()
    for buf in (
        env._drive_speed_sum,
        env._drive_speed_max,
        env._steer_speed_sum,
        env._steer_speed_max,
        env._motor_speed_count,
    ):
        buf[env_ids] = 0.0
    step = int(env.common_step_counter)
    if step - getattr(env, "_last_motor_print", -(10**9)) >= print_interval:
        env._last_motor_print = step
        print(
            f"[MotorSpeed] step {step}: drive mean/max = {drive_mean:.2f}/{drive_max:.2f} rad/s"
            f" | steer mean/max = {steer_mean:.2f}/{steer_max:.2f} rad/s"
        )


# ---------------------------------------------------------------------------
# Domain-randomization ramp curriculum
# ---------------------------------------------------------------------------


def _lerp(a: float, base: float, target: float) -> float:
    return base + a * (target - base)


def _lerp_range(a: float, base: tuple, target: tuple) -> tuple:
    return (_lerp(a, base[0], target[0]), _lerp(a, base[1], target[1]))


def _obs_noise_terms(env: ManagerBasedRLEnv):
    """Yield the GaussianNoiseCfg objects of the balance ``imu`` observation term."""
    obs_man = getattr(env, "observation_manager", None)
    if obs_man is not None:
        names_by_group = getattr(obs_man, "_group_obs_term_names", {})
        cfgs_by_group = getattr(obs_man, "_group_obs_term_cfgs", {})
        for g, names in names_by_group.items():
            if "imu" in names:
                cfg = cfgs_by_group[g][names.index("imu")]
                noise = getattr(cfg, "noise", None)
                if noise is not None and hasattr(noise, "std"):
                    yield noise
    obs = getattr(env.cfg, "observations", None)
    groups = obs.values() if isinstance(obs, dict) else ([getattr(obs, "policy", obs)] if obs else [])
    for group in groups:
        imu = getattr(group, "imu", None)
        noise = getattr(imu, "noise", None) if imu is not None else None
        if noise is not None and hasattr(noise, "std"):
            yield noise


def _get_obs_noise_std(env: ManagerBasedRLEnv, default: float = 0.0) -> float:
    for noise in _obs_noise_terms(env):
        return float(noise.std)
    return default


def _set_obs_noise_std(env: ManagerBasedRLEnv, std: float) -> None:
    for noise in _obs_noise_terms(env):
        noise.std = std


def ramp_domain_randomization(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    start_step: int = 0,
    end_step: int = 50000,
) -> float:
    """Ramp every ramp-able DR term from OFF (nominal) to its authored full range.

    ``alpha`` goes 0 -> 1 linearly from ``start_step`` to ``end_step`` (env-step
    units, ``env.common_step_counter``); each term is interpolated from its
    nominal value (scales -> 1.0, offsets/std/latency -> 0) to the value authored
    in the config, which is snapshotted once on the first call. Returns ``alpha``
    and prints it every ~2000 steps (skrl doesn't forward env logs to TensorBoard).

    Ground friction is NOT ramped (a shared USD material is unreliable to change
    mid-sim); it's pinned to a fixed nominal and the per-env robot tire material
    carries the friction DR instead.
    """
    step = int(env.common_step_counter)
    a = 1.0 if end_step <= start_step else (step - start_step) / (end_step - start_step)
    a = max(0.0, min(1.0, a))

    ev = env.event_manager.cfg

    def term(name):
        return getattr(ev, name, None)

    # snapshot authored "full" values once, before we start overwriting them
    if not hasattr(env, "_dr_full"):
        full = {}
        if (t := term("randomize_mass")) is not None:
            full["mass"] = tuple(t.params["mass_distribution_params"])
        if (t := term("randomize_com")) is not None:
            full["com"] = {k: tuple(v) for k, v in t.params["com_range"].items()}
        if (t := term("randomize_material")) is not None:
            full["mat_s"] = tuple(t.params["static_friction_range"])
            full["mat_d"] = tuple(t.params["dynamic_friction_range"])
        if (t := term("randomize_actuator_gains")) is not None:
            full["gain_k"] = tuple(t.params["stiffness_distribution_params"])
            full["gain_d"] = tuple(t.params["damping_distribution_params"])
        if (t := term("initial_state")) is not None:
            full["roll"] = tuple(t.params["pose_range"]["roll"])
            full["rollrate"] = tuple(t.params["velocity_range"]["roll"])
        if (t := term("reset_imu_sensor_model")) is not None:
            full["gyro"] = float(t.params["gyro_bias_std"])
            full["tilt"] = float(t.params["tilt_bias_std"])
            full["dmin"] = int(t.params["min_delay_steps"])
            full["dmax"] = int(t.params["max_delay_steps"])
        full["noise"] = _get_obs_noise_std(env)
        env._dr_full = full
    full = env._dr_full

    # scale-type terms collapse to (1,1); offsets / std / latency to 0
    if (t := term("randomize_mass")) is not None and "mass" in full:
        t.params["mass_distribution_params"] = _lerp_range(a, (1.0, 1.0), full["mass"])
    if (t := term("randomize_material")) is not None and "mat_s" in full:
        t.params["static_friction_range"] = _lerp_range(a, (1.0, 1.0), full["mat_s"])
        t.params["dynamic_friction_range"] = _lerp_range(a, (1.0, 1.0), full["mat_d"])
    if (t := term("randomize_actuator_gains")) is not None and "gain_k" in full:
        t.params["stiffness_distribution_params"] = _lerp_range(a, (1.0, 1.0), full["gain_k"])
        t.params["damping_distribution_params"] = _lerp_range(a, (1.0, 1.0), full["gain_d"])
    if (t := term("randomize_com")) is not None and "com" in full:
        com = t.params["com_range"]
        for k in com:
            com[k] = _lerp_range(a, (0.0, 0.0), full["com"][k])
    if (t := term("initial_state")) is not None and "roll" in full:
        t.params["pose_range"]["roll"] = _lerp_range(a, (0.0, 0.0), full["roll"])
        t.params["velocity_range"]["roll"] = _lerp_range(a, (0.0, 0.0), full["rollrate"])
    if (t := term("reset_imu_sensor_model")) is not None and "gyro" in full:
        t.params["gyro_bias_std"] = _lerp(a, 0.0, full["gyro"])
        t.params["tilt_bias_std"] = _lerp(a, 0.0, full["tilt"])
        t.params["min_delay_steps"] = int(round(_lerp(a, 0.0, full["dmin"])))
        t.params["max_delay_steps"] = int(round(_lerp(a, 0.0, full["dmax"])))
    _set_obs_noise_std(env, _lerp(a, 0.0, full["noise"]))

    if step - getattr(env, "_last_dr_print", -(10**9)) >= 2000:
        env._last_dr_print = step
        print(f"[DR ramp] step {step}: alpha = {a:.3f}")
    return a
