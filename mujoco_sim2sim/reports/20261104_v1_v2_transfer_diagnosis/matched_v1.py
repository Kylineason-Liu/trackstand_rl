"""V1 policy/torque law on exact v2 DR plants, initial states and noise streams."""

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import torch

REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[2]
sys.path.insert(0, str(ROOT / "mujoco_sim2sim/src"))
from newbike_mujoco.env import BikeBalanceEnv as V1Env
from newbike_mujoco.policy import DeployedPolicy as V1Policy
from newbike_mujoco_v2.model import build_env_from_mjcf
from newbike_mujoco_v2.viewer_envs import ViewerEnvGroup
from newbike_mujoco_v2 import contract as c


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics_hz", type=int, choices=(800, 1200), required=True)
    args = parser.parse_args()
    output = REPORT / f"matched_v1_{args.physics_hz}"
    output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    checkpoint = ROOT / "isaaclab_train/logs/rsl_rl/trackstand_rl_rsl/2026-10-21_11-18-29/model_2500.pt"
    policy = V1Policy(pt_path=checkpoint)
    source, _ = build_env_from_mjcf(c.DEFAULT_MJCF, seed=42, enable_dr=True, physics_hz=args.physics_hz)
    group = ViewerEnvGroup(source, 64, 42)
    obs = group.reset()
    envs, initial = [], []
    for index, src in enumerate(group.envs):
        # Clone the already randomized, verified v2 plant. V1's own DR is off
        # to avoid drawing a different mass/CoM/friction or gain realization.
        env = V1Env(model=src.model, seed=42 + index, enable_dr=False, physics_hz=args.physics_hz)
        env.enabled_terms = {"time_out", "fallen"}
        env.reset()
        env.data.qpos[:] = src.data.qpos
        env.data.qvel[:] = src.data.qvel
        env._spawn_pos[:] = src._spawn_pos
        env._yaw0, env._yaw_prev = src._yaw0, src._yaw_prev
        env._steer_cmd = src._steer_cmd
        env._gyro_bias[:] = src._gyro_bias
        env._accel_bias[:] = src._accel_bias
        env.rng = copy.deepcopy(src.rng)
        mujoco.mj_forward(env.model, env.data)
        for field in ("body_mass", "body_inertia", "body_ipos", "geom_friction", "dof_damping", "dof_frictionloss"):
            np.testing.assert_array_equal(getattr(env.model, field), getattr(src.model, field))
        np.testing.assert_array_equal(env.data.qpos, src.data.qpos)
        np.testing.assert_array_equal(env.data.qvel, src.data.qvel)
        np.testing.assert_array_equal(env.kp, src.kp)
        # Steering gains match. Rear law is deliberately v1 feedforward torque.
        assert env.kd[0] == src.kd[0] == .2 and env.kd[1] == 0.
        initial.append(dict(env_id=index, sampled_dr=src.dr, qpos=src.data.qpos.copy(),
                            qvel=src.data.qvel.copy(), observation=obs[index].copy()))
        envs.append(env)
    active = np.ones(64, dtype=bool)
    records, observations, actions, masks = [], [], [], []
    for step in range(1500):
        if not active.any():
            break
        action = np.zeros((64, 2), dtype=np.float32)
        action[active] = policy.act_batch(obs[active])
        observations.append(np.where(active[:, None], obs, np.nan))
        actions.append(np.where(active[:, None], action, np.nan))
        masks.append(active.copy())
        for index in np.flatnonzero(active):
            obs[index], done = envs[index].step(action[index], observe_terminal=False)
            if done:
                records.append(dict(env_id=int(index), **asdict(envs[index].stats)))
                active[index] = False
    assert len(records) == 64 and not active.any()
    records.sort(key=lambda e: e["env_id"])
    lengths = np.array([e["length"] for e in records])
    timeouts = sum(e["timeout"] for e in records)
    summary = dict(policy=str(checkpoint), policy_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   physics_pd_hz=args.physics_hz, policy_hz=50, completed=64,
                   timeouts=timeouts, fallen=64-timeouts, mean_s=float(lengths.mean()*.02),
                   median_s=float(np.median(lengths)*.02),
                   reference_plant="Exact v2 DR reset plant/state/IMU residuals/noise stream; fixed gains",
                   interpretation="Policy and rear action/controller differ together; not a controller-only ablation",
                   standard_v1_gain_randomization_applied=False, production_source_modified=False)
    np.savez_compressed(output / "policy_samples.npz", observation=np.asarray(observations),
                        raw_action=np.asarray(actions), active_mask=np.asarray(masks))
    (output / "initial_states.json").write_text(json.dumps(initial, default=serial, indent=2)+"\n")
    (output / "episodes.json").write_text(json.dumps(records, default=serial, indent=2)+"\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
