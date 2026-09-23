#!/usr/bin/env python3
"""Deployment check: run the 09090950 model_500 policy on the MuJoCo NewBike.

Mirrors the play_eval protocol: N episodes of up to 1500 steps (30 s at 50 Hz),
per-episode reset like the sim's initial_state event, and a summary in
play_eval-comparable form (epi len, timeout %, roll RMS, max roll, drift, yaw,
turn, death breakdown).

Usage:
    python scripts/run_deploy_eval.py --episodes 64 --seed 42
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from newbike_mujoco import BikeBalanceEnv, DeployedPolicy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    # load the model XML directly (like run_free_fall_viewer.py): it already
    # carries its own ground and lighting, so no scene wrapper is needed
    ap.add_argument("--scene", default=os.path.join(ROOT, "models/newbike/model.xml"))
    ap.add_argument("--onnx", default=os.path.join(ROOT, "policies/model_500.onnx"))
    ap.add_argument("--pt", default=os.path.join(ROOT, "policies/model_500.pt"))
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--steps", type=int, default=BikeBalanceEnv.MAX_STEPS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base_inertia", type=float, default=1.0,
                    help="Scale the base link's rotational inertia (default 1 = the USD "
                         "value: the fall matches the sim's full-gravity rate). Use e.g. "
                         "3000 for the long-survival closed-loop calibration (the fall "
                         "then looks like slow motion -- see the README's trade-off).")
    args = ap.parse_args()

    policy = DeployedPolicy(onnx_path=args.onnx, pt_path=args.pt)
    print(f"[INFO] policy backend: {policy.backend}")
    env = BikeBalanceEnv(args.scene, seed=args.seed)
    if args.base_inertia != 1.0:
        import mujoco
        m = env.model
        m.body_inertia[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base_link")] *= args.base_inertia
        print(f"[INFO] base inertia scaled x{args.base_inertia}")
    print(f"[INFO] scene: {args.scene}")
    print(f"[INFO] episodes={args.episodes} steps/ep={args.steps}")

    ep_lens, timeouts, deaths = [], 0, {}
    roll_rms, roll_maxs, drift_maxs, yaw_maxs, turns = [], [], [], [], []
    t0 = time.time()
    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.act(obs)
            obs, done = env.step(action)
        s = env.stats
        ep_lens.append(s.length)
        timeouts += int(s.timeout)
        if s.death:
            deaths[s.death] = deaths.get(s.death, 0) + 1
        roll_rms.append(np.sqrt(s.roll_sq / max(s.length, 1)))
        roll_maxs.append(s.roll_max)
        drift_maxs.append(s.drift_max)
        yaw_maxs.append(s.yaw_max)
        turns.append(s.turn)
        if (ep + 1) % 16 == 0:
            print(f"  episode {ep + 1}/{args.episodes}  ({(time.time() - t0) / (ep + 1):.2f} s/ep)")

    n = args.episodes
    print("\n===== DEPLOYMENT EVAL SUMMARY (MuJoCo) =====")
    print(f"  mean_episode_length    : {np.mean(ep_lens):9.1f} steps  ({np.mean(ep_lens) * 0.02:6.2f} s)")
    print(f"  min/max episode length : {np.min(ep_lens):9.1f} / {np.max(ep_lens):9.1f}")
    print(f"  timeout_rate (survived): {timeouts / n:9.3f}   (reached the horizon vs fell)")
    print(f"    time_out             : {timeouts / n * 100:8.1f}%")
    for k, v in sorted(deaths.items(), key=lambda kv: -kv[1]):
        print(f"    {k:20s}: {v / n * 100:8.1f}%")
    print(f"  roll_RMS               : {np.degrees(np.mean(roll_rms)):9.2f} deg  (tightness; lower = steadier)")
    print(f"  max_roll_angle         : {np.degrees(np.mean(roll_maxs)):9.2f} deg")
    print(f"  mean_worst_lean/episode: {np.degrees(np.mean(roll_maxs)):9.2f} deg")
    print(f"  drift mean/max         : {np.mean(drift_maxs):9.3f} m")
    print(f"  yaw mean/max           : {np.mean(yaw_maxs):9.2f} deg (heading vs spawn)")
    print(f"  turn/episode mean      : {np.mean(turns):9.1f} deg (360 deg = full circle)")
    print(f"  wall time              : {time.time() - t0:9.1f} s")


if __name__ == "__main__":
    main()
