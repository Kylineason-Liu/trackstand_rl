#!/usr/bin/env python3
"""Run the BACKUP (sensor-based) environment headlessly with a policy.

Uses THIS directory's env.py + .trackstand_deploy_augmented.xml (the
pre-refactor sensor versions) — the live files are not touched. The
live package is only imported for DeployedPolicy (the policy loader).

Usage:
    python run_backup_headless.py --steps 600 --seed 42
"""
import argparse
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/lmbike/lmbike_cyclerl-mod/mujoco_sim2sim/src")
from newbike_mujoco import DeployedPolicy  # noqa: E402  (policy loader only)

_loader = importlib.util.spec_from_file_location("backup_env", os.path.join(HERE, "env.py"))
backup_env = importlib.util.module_from_spec(_loader)
_loader.loader.exec_module(backup_env)

DEFAULT_ONNX = ("/home/lmbike/lmbike_cyclerl-mod/isaaclab_train/scripts/deployment/"
                "trackstand_rl_rsl_2026-09-17_19-46-37_model_41050.onnx")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600, help="control steps to run (0.02 s each)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--onnx", default=DEFAULT_ONNX)
    args = ap.parse_args()

    env = backup_env.BikeBalanceEnv(
        scene_path=os.path.join(HERE, ".trackstand_deploy_augmented.xml"), seed=args.seed)
    env.reset_ranges = ((0.0, 0.0), (0.0, 0.0))
    env.enabled_terms = {"time_out", "fallen"}
    policy = DeployedPolicy(onnx_path=args.onnx, pt_path=None)
    print(f"scene sensors: {env.model.nsensor} "
          f"(gyro id {env._imu_sid}, vel id {env._vel_sid})")
    print(f"policy backend: {policy.backend} | {args.steps} steps, seed {args.seed}")

    obs = env.reset()
    falls, n, ep = 0, 0, 1
    while n < args.steps:
        obs, done = env.step(policy.act(obs))
        n += 1
        if done:
            s = env.stats
            print(f"  episode {ep}: {s.length} steps, {s.death or 'timeout'}")
            falls += int(s.death is not None)
            ep += 1
            obs = env.reset()
    print(f"{n} steps total, {falls} fall(s) — sensor observation path")
    print(f"final sensordata (gyro, vel): "
          f"{np.asarray(env.data.sensordata).round(4)}")


if __name__ == "__main__":
    main()
