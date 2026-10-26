"""Validate saved final MuJoCo captures and replay identical Isaac raw actions.

The replay isolates plant/integration response from closed-loop policy drift.
It does not retune models or imply engine-difference causes have been isolated.
"""

import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[2]
sys.path.insert(0, str(REPO / "mujoco_sim2sim/src"))
from newbike_mujoco import contract as c
from newbike_mujoco.env import _roll_from_quat
from newbike_mujoco.model import build_env_from_mjcf


def validate(folder):
    path, = folder.glob("*_evaluation.npz")
    summary, = folder.glob("*_summary.json")
    metadata, = folder.glob("*_metadata.json")
    results = json.loads(summary.read_text())
    meta = json.loads(metadata.read_text())
    assert meta["completed"] and meta["completed_policy_steps"] == 3000
    with np.load(path) as z:
        obs, raw, torques = z["observation"], z["raw_action"], z["physics_motor_torque_nm"]
        target, term = z["held_steering_target_rad"], z["termination_flags"]
        assert obs.shape == (3000, 64, 11) and raw.shape == (3000, 64, 2)
        assert torques.shape == (3000, 64, 8, 2)
        assert all(np.isfinite(z[name]).all() for name in (
            "observation", "raw_action", "held_steering_target_rad", "position_rad", "velocity_rad_s",
            "physics_motor_torque_nm", "reward"))
        assert np.max(np.abs(raw)) <= 1.
        assert np.max(np.abs(target)) <= 1.57+1e-6
        assert np.max(np.abs(torques[..., 0])) <= 5.6+1e-6
        assert np.max(np.abs(torques[..., 1])) <= 6.28+1e-6
        expected_rear = (raw[..., 1]*np.float32(6.28))[:, :, None]
        assert np.max(np.abs(torques[..., 1]-expected_rear)) < 1e-6
        assert all(results["termination_counts"][name] == int(term[..., i].sum())
                   for i, name in enumerate(c.TERMINATIONS))
        histogram, = folder.glob("*_epi_len_histogram.csv")
        assert np.loadtxt(histogram, delimiter=",", skiprows=1)[:, 2].sum() == results["completed_episodes"]
        # Reconstruct50Hz scale/clip/slew, respecting every independent reset.
        previous = np.zeros(64)
        max_slew_error = 0.
        for step in range(3000):
            requested = np.clip(raw[step, :, 0]*np.float32(1.57), -1.57, 1.57)
            expected = previous+np.clip(requested-previous, -.478, .478)
            max_slew_error = max(max_slew_error, float(np.max(np.abs(expected-target[step]))))
            previous = target[step].copy()
            previous[np.any(term[step], axis=1)] = 0.
        assert max_slew_error < 1e-6
        results["capture_validation"] = dict(
            policy_frames=3000*64, motor_physics_frames=3000*64*8,
            raw_min=raw.min(axis=(0, 1)).tolist(), raw_max=raw.max(axis=(0, 1)).tolist(),
            max_abs_actuator_torque_nm=np.max(np.abs(torques), axis=(0, 1, 2)).tolist(),
            max_abs_steering_target_rad=float(np.max(np.abs(target))),
            max_reconstructed_slew_error_rad=max_slew_error,
            observation_action_reward_torque_finite=True,
            input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    return results


def replay():
    source = REPO / "isaaclab_train/reports/20261021_75300_actuation/no_dr/motor_samples.npz"
    steps = 25  # .5s, fixed actions; no automatic resets in this diagnostic.
    with np.load(source) as z:
        raw = z["raw_action"][:steps, 0].copy()
        isaac_position = z["position_after_rad"][:steps, :, 0].reshape(-1, 2)
        isaac_torque = z["submitted_motor_torque_nm"][:steps, :, 0].reshape(-1, 2)
        isaac_target = z["held_steering_target_rad"][:steps, :, 0].reshape(-1)
        isaac_roll = z["body_roll_drift_yaw"][:steps, :, 0, 0].reshape(-1)
    env, _ = build_env_from_mjcf(c.DEFAULT_MJCF, 42)
    env.reset()
    position, torque, roll, target = [], [], [], []
    advance = mujoco.mj_step

    def record(model, data):
        torque.append(data.ctrl[env._acts].copy())
        target.append(env._steer_cmd)
        advance(model, data)
        position.append(data.qpos[[env._steer_qadr, env._rear_qadr]].copy())
        roll.append(_roll_from_quat(env._base_quat()))

    with patch("newbike_mujoco.env.mujoco.mj_step", side_effect=record):
        for action in raw:
            env.step(action, observe_terminal=False)
    position, torque, target, roll = map(np.asarray, (position, torque, target, roll))
    time = (np.arange(len(position))+1)*c.PHYSICS_DT
    np.savez_compressed(OUT / "matched_action_replay.npz", time_s=time, raw_action=raw,
                        isaac_position_rad=isaac_position, mujoco_position_rad=position,
                        isaac_motor_torque_nm=isaac_torque, mujoco_motor_torque_nm=torque,
                        isaac_target_rad=isaac_target, mujoco_target_rad=target,
                        isaac_roll_rad=isaac_roll, mujoco_roll_rad=roll)
    target_error = float(np.max(np.abs(isaac_target-target)))
    rear_error = float(np.max(np.abs(isaac_torque[:, 1]-torque[:, 1])))
    assert target_error < 1e-6 and rear_error < 1e-6
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    axes[0].plot(time, np.degrees(isaac_target), color="black", ls="--", label="Same held position target")
    axes[0].plot(time, np.degrees(isaac_position[:, 0]), label="Isaac actual steering")
    axes[0].plot(time, np.degrees(position[:, 0]), label="MuJoCo actual steering")
    axes[0].set_ylabel("Steering (deg)")
    axes[1].plot(time, isaac_torque[:, 0], label="Isaac steering actuator torque")
    axes[1].plot(time, torque[:, 0], label="MuJoCo steering actuator torque")
    axes[1].plot(time, isaac_torque[:, 1], color="black", ls="--", label="Same rear actuator torque")
    axes[1].set_ylabel("Actuator torque (Nm)")
    axes[2].plot(time, np.degrees(isaac_roll), label="Isaac actual roll")
    axes[2].plot(time, np.degrees(roll), label="MuJoCo actual roll")
    axes[2].axhline(30., color="red", ls=":")
    axes[2].axhline(-30., color="red", ls=":")
    axes[2].set_ylabel("Roll (deg)")
    axes[2].set_xlabel("Time after spawn (s),400Hz samples")
    axes[0].set_title("Matched raw-action replay — model75300,noDR,env0\n"
                      "Same requests, different plant response; no automatic reset during replay")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=9)
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"matched_action_replay.{ext}", dpi=160)
    plt.close(fig)
    return dict(steps=steps, duration_s=steps*c.CONTROL_DT, automatic_resets=False,
                first_physics_step_isaac_torque_nm=isaac_torque[0].tolist(),
                first_physics_step_mujoco_torque_nm=torque[0].tolist(),
                first_physics_step_isaac_position_rad=isaac_position[0].tolist(),
                first_physics_step_mujoco_position_rad=position[0].tolist(),
                max_held_target_difference_rad=target_error, max_rear_motor_request_difference_nm=rear_error,
                isaac_roll_peak_deg=float(np.max(np.abs(np.degrees(isaac_roll)))),
                mujoco_roll_peak_deg=float(np.max(np.abs(np.degrees(roll)))),
                note="Different physical states lead to different PD feedback torques; no causal engine ablation")


if __name__ == "__main__":
    results = dict(final_no_dr=validate(OUT / "no_dr"), final_dr=validate(OUT / "dr"),
                   matched_action_replay=replay())
    (OUT / "validation_results.json").write_text(json.dumps(results, indent=2, allow_nan=False)+"\n")
    print(json.dumps(results, indent=2))
