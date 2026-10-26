"""Add read-only physics-substep recording around the production DR evaluator.

No extra observations, resets, RNG draws or state modifications. Original CLI
captures are retained for byte comparisons with the October 23 evaluations.
"""

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "mujoco_sim2sim/src"))
from newbike_mujoco.env import BikeBalanceEnv
from newbike_mujoco.viewer_envs import ViewerEnvGroup

SCRIPT = ROOT / "mujoco_sim2sim/scripts/run_deploy_eval.py"
CHECKPOINT = ROOT / "isaaclab_train/logs/rsl_rl/trackstand_rl_rsl/2026-10-21_11-18-29/model_2500.pt"
PHYSICS_FIELDS = ("position_before_rad", "velocity_before_rad_s", "position_after_rad",
                  "velocity_after_rad_s", "computed_motor_torque_nm", "submitted_motor_torque_nm",
                  "actual_motor_torque_nm", "controller_kp", "controller_kd")


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics_hz", type=int, choices=(400, 800), required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--num_steps", type=int, default=3000)
    parser.add_argument("--num_envs", type=int, default=64)
    args = parser.parse_args()
    output = args.capture_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    count, substeps = args.num_envs, args.physics_hz // 50
    shape = (args.num_steps, substeps, count, 2)
    arrays = {key: np.empty(shape) for key in PHYSICS_FIELDS}
    held = np.empty(shape[:-1])
    counters = np.zeros(count, dtype=int)
    draws, group_ref = [], []
    state = {"active": False, "step": 0, "sub": np.zeros(count, dtype=int)}
    original_init, original_reset = ViewerEnvGroup.__init__, BikeBalanceEnv.reset
    original_group_step, original_physics_step = ViewerEnvGroup.step, mujoco.mj_step
    by_data = {}

    def initialize(group, *positional, **keywords):
        original_init(group, *positional, **keywords)
        group_ref.append(group)
        for index, env in enumerate(group.envs):
            env._capture_id = index
            by_data[id(env.data)] = env

    def reset(env):
        observation = original_reset(env)
        index = env._capture_id
        counters[index] += 1
        draws.append(dict(env=index, episode=int(counters[index]), dr=env.dr,
                          kp=env.kp.copy(), kd=env.kd.copy(),
                          qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy()))
        return observation

    def physics_step(model, data):
        if not state["active"]:
            return original_physics_step(model, data)
        env = by_data[id(data)]
        index = env._capture_id
        slot = state["step"], state["sub"][index], index
        assert slot[0] < args.num_steps and slot[1] < substeps
        qids = [env._steer_qadr, env._rear_qadr]
        arrays["position_before_rad"][slot] = data.qpos[qids]
        arrays["velocity_before_rad_s"][slot] = data.qvel[env._dofs]
        arrays["computed_motor_torque_nm"][slot] = env.computed_torque
        arrays["submitted_motor_torque_nm"][slot] = data.ctrl[env._acts]
        arrays["controller_kp"][slot], arrays["controller_kd"][slot] = env.kp, env.kd
        held[slot] = env._steer_cmd
        original_physics_step(model, data)
        arrays["position_after_rad"][slot] = data.qpos[qids]
        arrays["velocity_after_rad_s"][slot] = data.qvel[env._dofs]
        arrays["actual_motor_torque_nm"][slot] = data.actuator_force[env._acts]
        state["sub"][index] += 1

    def group_step(group, actions):
        state["sub"][:] = 0
        state["active"] = True
        try:
            result = original_group_step(group, actions)
        finally:
            state["active"] = False
        assert (state["sub"] == substeps).all()
        state["step"] += 1
        return result

    argv = [str(SCRIPT), "--pt", str(CHECKPOINT), "--dr", "--num_envs", str(count),
            "--num_steps", str(args.num_steps), "--seed", "42", "--physics_hz", str(args.physics_hz),
            "--output_dir", str(output)]
    print("Production invocation:", " ".join(argv), flush=True)
    with patch.object(ViewerEnvGroup, "__init__", initialize), patch.object(BikeBalanceEnv, "reset", reset), \
            patch.object(ViewerEnvGroup, "step", group_step), patch.object(mujoco, "mj_step", physics_step), \
            patch.object(sys, "argv", argv):
        scope = runpy.run_path(str(SCRIPT), run_name="capture_production_evaluator")
        scope["main"]()
    assert len(group_ref) == 1 and state["step"] == args.num_steps
    assert all(np.isfinite(value).all() for value in arrays.values())
    np.savez_compressed(output / "physics_samples.npz", **arrays, held_steering_target_rad=held)
    (output / "dr_draws.json").write_text(json.dumps(draws, indent=2, default=serial, allow_nan=False) + "\n")
    metadata = dict(capture_completed=True, checkpoint_source=str(CHECKPOINT),
                    checkpoint_sha256=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
                    production_invocation=argv, policy_dt_s=.02, physics_dt_s=1 / args.physics_hz,
                    decimation=substeps, num_envs=count, completed_policy_steps=args.num_steps,
                    dr=True, training_dr=True, pure_survival=False, seed=42, mujoco_version=mujoco.__version__,
                    motor_order=["steering", "rear"],
                    velocity_after_meaning="Native post-integration velocity BEFORE production speed projection",
                    torque_meaning="data.actuator_force: applied motor actuator force, NOT total net joint torque",
                    timing="Commands at physics-step start; responses at end, before projection and automatic reset")
    (output / "experiment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Complete read-only physics capture:", output, flush=True)


if __name__ == "__main__":
    main()
