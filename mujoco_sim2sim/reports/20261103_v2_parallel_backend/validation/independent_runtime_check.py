"""Headless controlled engine checks; run only after coordinator allocation."""

import ast
from collections.abc import Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/trackstand_v2_validation_mpl")

import mujoco
import numpy as np
import torch

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
sys.path.insert(0, str(ROOT / "mujoco_sim2sim/src"))
from newbike_mujoco_v2 import contract as c
from newbike_mujoco_v2.env import BikeBalanceEnv
from newbike_mujoco_v2.model import load_trackstand_model


def reference_class():
    path = ROOT / (
        "isaaclab_train/source/trackstand_rl/trackstand_rl/tasks/manager_based/trackstand/"
        "mdp/stationary/velocity_pi.py"
    )
    cls = next(node for node in ast.parse(path.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == "VelocityIntegral")
    namespace = {"torch": torch, "math": math, "Sequence": Sequence}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["VelocityIntegral"]


def main():
    torch.set_num_threads(1)
    model = load_trackstand_model(c.DEFAULT_MJCF)
    env = BikeBalanceEnv(model=model, seed=42)
    env.OBS_NOISE_STD = (0.0,) * 11  # Controlled algebra check only; learned eval retains noise.
    cases = []
    np.testing.assert_allclose(env.model.dof_damping[env._dofs],
                               [0.000910713492, 0.0], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(env.model.dof_frictionloss[env._dofs],
                               [0.046036056104, 0.106], rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(env.model.actuator_gainprm[env._acts, 0], [1.0, 1.0])
    assert env.decimation == 24 and env.integral_decimation == 6
    assert env.physics_dt == 1 / 1200 and env.integral_dt == 0.005
    cases.append("Motor gains, final effort limits, passive friction, 50/200/1200-Hz clocks")

    reference = reference_class()(torch.zeros((1, 1), dtype=torch.float64), 3.3)
    env.reset()
    original_step = mujoco.mj_step
    original_integral_update = env.integral.update
    observed, integral_ticks, input_targets = [], [], []

    def captured_integral(error, dt, p_gain, limit):
        reference.update(torch.tensor([[error]], dtype=torch.float64), dt,
                         torch.tensor([[p_gain]], dtype=torch.float64),
                         torch.tensor([[limit]], dtype=torch.float64))
        value = original_integral_update(error, dt, p_gain, limit)
        assert value == reference.torque_nm.item()
        assert env.integral.error_rad == reference.error_rad.item()
        assert env.integral.previous_error == reference.previous_error.item()
        integral_ticks.append(env._physics_steps)
        return value

    def captured_step(model_arg, data_arg):
        error = env._drive_cmd - data_arg.qvel[env._rear_dof]
        expected = np.clip(0.3 * error + env.integral.torque_nm, -6.28, 6.28)
        assert abs(data_arg.ctrl[env._dr_drive_act] - expected) < 1e-12
        expected_steer = np.clip(10.0 * (env._steer_cmd - data_arg.qpos[env._steer_qadr])
                                 - 0.2 * data_arg.qvel[env._steer_dof], -5.6, 5.6)
        assert abs(data_arg.ctrl[env._dr_steer_act] - expected_steer) < 1e-12
        observed.append((env._physics_steps - 1, env._drive_cmd, error,
                         env.integral.torque_nm, expected, data_arg.qpos[env._rear_qadr]))
        return original_step(model_arg, data_arg)

    env.integral.update = captured_integral
    mujoco.mj_step = captured_step
    try:
        targets = [(0.0, 0.01), (0.05, 0.02), (-0.05, -0.01), (0.0, 0.0),
                   (1.0, 1.0), (-1.0, -1.0), (0.0, 0.0), (0.2, -0.02)]
        previous_steer = 0.0
        for action in targets:
            input_targets.append(action)
            env.step(action)
            target = float(np.clip(np.float32(action[1]) * np.float32(61.6), -61.6, 61.6))
            assert env._drive_cmd == target
            assert abs(env._steer_cmd - previous_steer) <= 23.9 * 0.02 + 1e-15
            assert abs(env._steer_cmd) <= 1.57
            assert abs(env.wheel_net_angle - env.data.qpos[env._rear_qadr]) < 1e-10
            assert np.isfinite(env.observe()).all()
            previous_steer = env._steer_cmd
    finally:
        mujoco.mj_step = original_step
        env.integral.update = original_integral_update
    assert len(observed) == 8 * 24
    assert integral_ticks == list(range(0, 192, 6))
    assert abs(env.physics_torques[:, 0]).max() <= 5.6
    assert abs(env.physics_torques[:, 1]).max() <= 6.28
    cases.append("192 actual physics steps: submitted steer PD/rear P+I equations and 32 exact I ticks")
    env.reset()
    assert env.integral.torque_nm == env.integral.error_rad == env.integral.previous_error == 0.0
    assert not env.integral.initialized and env.integral.limited_updates == 0
    assert env._drive_cmd == env.wheel_net_angle == env._previous_rear_position == 0.0
    assert env._steer_cmd == 0.0 and env._physics_steps == 0
    assert np.all(env.data.ctrl == 0.0)
    cases.append("Episode reset clears physical motor targets, I/error/history, wheel-net and zero references")

    # Form a known body orientation independently from roll/pitch/yaw matrices.
    roll, pitch, yaw = 0.13, -0.09, 0.27
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    rotation = (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
                @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
                @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, rotation.ravel())
    env.data.qpos[env._base_qposadr + 3:env._base_qposadr + 7] = quat
    env.data.qpos[[env._steer_qadr, env._rear_qadr]] = [0.31, 13.0]
    env.data.qvel[env._base_qveladr + 3:env._base_qveladr + 6] = [0.1, -0.2, 0.3]
    env.data.qvel[env._dofs] = [0.4, -0.7]
    mujoco.mj_forward(env.model, env.data)
    gravity = rotation.T @ [0, 0, -1]
    expected = np.array([gravity[1], gravity[2], 0.1, -0.2, 0.3, 0.31,
                         (13.0 + 2 * np.pi) % (4 * np.pi) - 2 * np.pi,
                         0.4, -0.7, np.sin(0.31), np.cos(0.31)], dtype=np.float32)
    np.testing.assert_allclose(env.observe(), expected, rtol=0.0, atol=1e-7)
    cases.append("11-observation order/units/body rotation/native 4pi rear branch, with controlled zero noise")

    env.reset()
    env.data.qpos[env._rear_qadr] = 7.19
    env._previous_rear_position = 7.19
    env.wheel_net_angle = 7.19
    env.data.qvel[env._rear_dof] = 61.0
    mujoco.mj_forward(env.model, env.data)
    env.step([0.0, 1.0])
    assert env.wheel_net_angle > 7.2 and env.termination_flags["wheel_net"]
    cases.append("Wheel-net crossing terminates in the same policy interval, not one interval later")

    env.reset()
    env._step = 1499
    env.step([0.0, 0.0])
    assert env.termination_flags["time_out"] and env.stats.length == 1500
    cases.append("30-second timeout and explicit reset bookkeeping")

    invalid_cases = []
    for name in ("root_position", "joint_position", "zero_quaternion", "integral_error",
                 "integral_torque", "integral_previous_error", "wheel_net", "front_overspeed"):
        env.reset()
        if name == "root_position":
            env.data.qpos[env._base_qposadr] = np.nan
        elif name == "joint_position":
            env.data.qpos[env._steer_qadr] = np.nan
        elif name == "zero_quaternion":
            env.data.qpos[env._base_qposadr + 3:env._base_qposadr + 7] = 0.0
        elif name == "front_overspeed":
            env.data.qvel[env._front_dof] = 301.0
        elif name == "wheel_net":
            env.wheel_net_angle = np.nan
        else:
            setattr(env.integral, {"integral_error": "error_rad", "integral_torque": "torque_nm",
                                   "integral_previous_error": "previous_error"}[name], np.nan)
        with np.errstate(invalid="ignore"):
            env._check_terminations()
            env._compute_reward()
        assert env.termination_flags["diverged"], name
        assert np.isfinite(env.last_reward), name
        invalid_cases.append(name)
    cases.append("Eight controlled invalid-state checks: diverged and finite terminal reward, before solver stepping")

    randomized = BikeBalanceEnv(model=model, seed=42, enable_dr=True)
    draws = []
    for _ in range(20):
        randomized.reset()
        np.testing.assert_array_equal(randomized.kp, [10.0, 0.0])
        np.testing.assert_array_equal(randomized.kd, [0.2, 0.3])
        assert randomized.integral.torque_nm == 0.0
        draws.append({key: np.asarray(randomized.dr[key]).tolist()
                      for key in ("mass_scale", "com_offset", "tire_friction", "gyro_bias", "accel_bias")})
    assert draws[0] != draws[1]
    cases.append("20 independent DR resets retain fixed steering/rear gains while other DR draws change")
    np.savez_compressed(OUT / "runtime_control_samples.npz", samples=np.array(observed),
                        fields=np.array(["physics_step", "drive_target_rad_s", "error_rad_s", "I_nm",
                                         "expected_applied_drive_torque_nm", "pre_step_rear_position_rad"]),
                        input_policy_targets=np.array(input_targets), integral_ticks=np.array(integral_ticks))
    result = dict(passed=True, cases=cases, controlled_policy_steps=8,
                  captured_physics_steps=len(observed), captured_integral_updates=len(integral_ticks),
                  standalone_wheel_net_policy_steps=1, standalone_timeout_policy_steps=1,
                  dr_reset_count=len(draws), dr_draws=draws, invalid_state_cases=invalid_cases,
                  learned_policy_evaluation=False, gpu=False, hardware=False,
                  model_sha256=hashlib.sha256(c.DEFAULT_MJCF.read_bytes()).hexdigest(),
                  versions=dict(mujoco=mujoco.__version__, numpy=np.__version__, torch=torch.__version__))
    (OUT / "independent_runtime_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "dr_draws"}, indent=2))


if __name__ == "__main__":
    main()
