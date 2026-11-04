"""Validate captured references and document startup versus later balancing motion."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


meta = json.loads((OUT / "experiment.json").read_text())
status = json.loads((OUT / "execution_status.json").read_text())
assert meta["capture_completed"] and status["completed"]
z = np.load(OUT / "motor_samples.npz")
steps, decimation, motors = z["position_before_rad"].shape
assert (steps, decimation, motors) == (1500, 24, 2)
assert meta["physics_hz"] == 1200 and meta["policy_hz"] == 50 and meta["integral_hz"] == 200
seed = meta["arguments"]["seed"]
assert meta["dr"] is False and seed == 42
assert z["done"].shape == (1500, 64) and z["done"].sum() == 64
assert not z["done"][:-1].any() and z["done"][-1].all()
assert z["termination_flags"][-1].sum(0).tolist() == [64, 0, 0, 0, 0, 0]
assert np.isfinite(z["observation"]).all() and np.max(np.abs(z["raw_action"])) <= 1
initial_q = z["position_before_rad"][0, 0]
initial_v = z["velocity_before_rad_s"][0, 0]
assert np.array_equal(initial_q, [0, 0]) and np.array_equal(initial_v, [0, 0])
assert z["drive_integral_torque_nm"][0, 0] == 0
assert np.array_equal(z["held_target"], np.repeat(z["held_target"][:, :1], 24, axis=1))
held = z["held_target"]
scaled = z["raw_action"] * np.array([1.57, 61.6], dtype=np.float32)
assert np.array_equal(held[:, 0, 1], np.clip(scaled[:, 1], -61.6, 61.6))
expected = (z["controller_kp"] * (np.stack((held[:, :, 0], np.zeros_like(held[:, :, 0])), axis=-1)
                                 - z["position_before_rad"])
            + z["controller_kd"] * (np.stack((np.zeros_like(held[:, :, 1]), held[:, :, 1]), axis=-1)
                                     - z["velocity_before_rad_s"]))
expected[:, :, 1] += z["drive_integral_torque_nm"]
torque_error = float(np.max(np.abs(expected - z["computed_motor_torque_nm"])))
assert torque_error < 2e-6
clip_error = float(np.max(np.abs(np.clip(z["computed_motor_torque_nm"],
                                       np.array([-5.6, -6.28], dtype=np.float32),
                                       np.array([5.6, 6.28], dtype=np.float32)) - z["applied_motor_torque_nm"])))
submitted_error = float(np.max(np.abs(z["applied_motor_torque_nm"] - z["submitted_motor_torque_nm"])))
assert clip_error == 0 and submitted_error == 0
# I changes only at physics substeps 0,6,12,18; every tick has exact 6-substep hold.
integral_hold_error = float(np.max(np.abs(z["drive_integral_torque_nm"] - np.repeat(
    z["drive_integral_torque_nm"][:, ::6], 6, axis=1))))
assert integral_hold_error == 0
for label in ("env", "agent"):
    delta = json.loads((OUT / f"{label}_configuration_differences.json").read_text())
    allowed = {"log_dir", "scene.num_envs", "scene.terrain.num_envs"} if label == "env" else {"load_checkpoint"}
    assert all(item["path"] in allowed for item in delta)

policy_csv = list(csv.DictReader((OUT / "policy_targets_50hz.csv").open()))
velocity_csv = list(csv.DictReader((OUT / "joint_velocity_reference_1200hz.csv").open()))
physics_csv = list(csv.DictReader((OUT / "motor_physics_1200hz.csv").open()))
assert len(policy_csv) == 1500 and len(velocity_csv) == 36001 and len(physics_csv) == 36000
policy_t = np.arange(1500) * .02
physics_t = np.arange(36000) / 1200
response_t = (np.arange(36000) + 1) / 1200
state_t = np.arange(36001) / 1200
assert np.allclose([float(row["time_s"]) for row in policy_csv], policy_t, atol=1e-12, rtol=0)
assert np.allclose([float(row["time_s"]) for row in velocity_csv], state_t, atol=1e-12, rtol=0)
actual_velocity = np.vstack((initial_v, z["velocity_after_rad_s"].reshape(-1, 2)))
csv_velocity = np.array([[row["steering_velocity_rad_s"], row["drive_velocity_rad_s"]]
                         for row in velocity_csv], dtype=np.float32)
assert np.array_equal(csv_velocity, actual_velocity)
csv_targets = np.array([[row["held_steering_position_target_rad"], row["drive_velocity_target_rad_s"]]
                        for row in policy_csv], dtype=np.float32)
assert np.array_equal(csv_targets, held[:, 0])
assert np.allclose([float(row["command_time_s"]) for row in physics_csv], physics_t, atol=1e-12, rtol=0)
assert np.allclose([float(row["response_time_s"]) for row in physics_csv], response_t, atol=1e-12, rtol=0)

windows = {}
for start, stop in ((0, 1), (1, 2), (2, 5), (5, 10), (10, 15), (15, 20), (20, 30)):
    target = held[start * 50:stop * 50, 0]
    velocity = z["velocity_after_rad_s"][start * 50:stop * 50].reshape(-1, 2)
    windows[f"{start}_{stop}_s"] = dict(target_min=target.min(0).tolist(), target_max=target.max(0).tolist(),
                                        actual_velocity_rms_rad_s=np.sqrt(np.mean(velocity ** 2, axis=0)).tolist(),
                                        actual_velocity_peak_rad_s=np.max(np.abs(velocity), axis=0).tolist())

reference = dict(schema_version=1, task=meta["task"], model=Path(meta["checkpoint_source"]).name,
                 checkpoint_sha256=meta["checkpoint_sha256"], seed=seed, dr=False,
                 num_envs=64, environment_index=0, episode_index=1,
                 duration_s=30., policy_hz=50., physics_hz=1200., integral_hz=200.,
                 initial_native_joint_position_rad=dict(steering=float(initial_q[0]), drive=float(initial_q[1])),
                 initial_joint_velocity_rad_s=dict(steering=float(initial_v[0]), drive=float(initial_v[1])),
                 initial_integral_torque_nm=0.,
                 steering_target_reference="Isaac steering_joint position; reset initial coordinate recorded above",
                 velocity_reference="Isaac joint coordinates; physical direction requires verified hardware sign mapping",
                 policy_target_columns=list(policy_csv[0]), policy_target_samples=1500,
                 policy_target_first_time_s=0., policy_target_last_time_s=29.98, policy_target_coverage_end_s=30.,
                 state_reference_columns=list(velocity_csv[0]), state_reference_samples=36001,
                 state_reference_first_time_s=0., state_reference_last_time_s=30.,
                 replay_command_columns=["held_steering_position_target_rad", "drive_velocity_target_rad_s"],
                 replay_command_meanings=["postclip/post-slew steering position, rad", "held rear velocity target, rad/s"],
                 recommended_replay_start_s=0., recommended_replay_duration_s=10.,
                 recommendation_basis="0–1 s initial transient; 5–10 s lower-amplitude ongoing balancing, not zero motion",
                 window_statistics=windows,
                 files={name: dict(sha256=sha(OUT / name)) for name in (
                     "motor_samples.npz", "policy_targets_50hz.csv", "joint_velocity_reference_1200hz.csv",
                     "motor_physics_1200hz.csv", "experiment.json")})
(OUT / "reference_metadata.json").write_text(json.dumps(reference, indent=2) + "\n")


def graph(name, start, stop):
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    q_state = np.vstack((initial_q, z["position_after_rad"].reshape(-1, 2)))
    psel = (policy_t >= start) & (policy_t <= stop)
    state_sel = (state_t >= start) & (state_t <= stop)
    command_sel = (physics_t >= start) & (physics_t <= stop)
    axes[0].step(policy_t[psel], scaled[psel, 0], where="post", label="scaled raw steering action", alpha=.6)
    axes[0].step(policy_t[psel], held[psel, 0, 0], where="post", label="held steering position target")
    axes[0].plot(state_t[state_sel], q_state[state_sel, 0], label="actual steering position", linewidth=.9)
    axes[0].set_ylabel("Steering position (rad)")
    axes[1].step(policy_t[psel], held[psel, 0, 1], where="post", label="rear velocity target")
    axes[1].plot(state_t[state_sel], actual_velocity[state_sel, 1], label="actual rear velocity", linewidth=.9)
    axes[1].set_ylabel("Rear velocity (rad/s)")
    axes[2].plot(state_t[state_sel], actual_velocity[state_sel, 0], label="actual steering velocity; no steering velocity target")
    axes[2].set_ylabel("Steering velocity (rad/s)")
    axes[3].plot(physics_t[command_sel], z["drive_integral_torque_nm"].reshape(-1)[command_sel], label="rear integral torque")
    axes[3].plot(physics_t[command_sel], z["applied_motor_torque_nm"].reshape(-1, 2)[command_sel, 1], label="rear applied actuator torque")
    axes[3].set_ylabel("Rear torque (N m)")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(loc="lower right", fontsize=8)
        if start == 0:
            ax.axvline(0., color="red", linewidth=1., label="spawn")
        lines = [np.asarray(line.get_ydata(), dtype=float) for line in ax.lines if len(line.get_ydata()) > 2]
        values = np.concatenate(lines)
        span = max(float(np.ptp(values)), .01)
        ax.set_ylim(float(values.min()) - .10 * span, float(values.max()) + .10 * span)
        ax.text(.005, .97, f"min={values.min():.5g}, max={values.max():.5g}", transform=ax.transAxes,
                va="top", fontsize=8, bbox=dict(facecolor="white", alpha=.7, edgecolor="none"))
    axes[-1].set_xlabel("Time since episode spawn (s)")
    axes[-1].set_xlim(start, stop)
    fig.suptitle(f"IsaacLab v2 model_26600 | no DR | env0 episode1 | {start:g}–{stop:g} s\n"
                 "policy 50 Hz; physics/PD 1200 Hz; rear I 200 Hz; red line = spawn")
    fig.tight_layout(rect=(0, 0, 1, .95))
    for extension in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{name}.{extension}", dpi=160)
    plt.close(fig)


graph("capture_full_episode", 0, 30)
graph("capture_startup", 0, 2)
graph("capture_replay_first10s", 0, 10)
graph("capture_later_balancing", 5, 10)
checks = dict(passed=True, policy_samples=1500, physics_intervals=36000, actual_state_rows=36001,
              full_duration_s=30., first_episode_completed=True, completed_episodes_all_envs=64,
              termination_counts=dict(zip(meta["termination_names"], z["termination_flags"][-1].sum(0).tolist())),
              initial_native_joint_position_rad=initial_q.tolist(), initial_joint_velocity_rad_s=initial_v.tolist(),
              initial_integral_torque_nm=float(z["drive_integral_torque_nm"][0, 0]),
              target_hold_exact=True, CSV_velocity_and_targets_float32_exact=True,
              maximum_motor_equation_error_nm=torque_error, maximum_effort_clip_error_nm=clip_error,
              maximum_submitted_vs_applied_error_nm=submitted_error,
              maximum_integral_hold_error_nm=integral_hold_error,
              rear_integral_max_abs_nm=float(np.max(np.abs(z["drive_integral_torque_nm"]))),
              physics_actual_velocity_peak_rad_s=np.max(np.abs(z["velocity_after_rad_s"]), axis=(0, 1)).tolist(),
              all_original_inputs_preserved=json.loads((OUT / "input_manifest_before.json").read_text()) ==
                  json.loads((OUT / "input_manifest_after.json").read_text()))
assert checks["all_original_inputs_preserved"]
(OUT / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")
print(json.dumps(checks, indent=2))
