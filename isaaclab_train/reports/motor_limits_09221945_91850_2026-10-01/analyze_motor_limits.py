"""Summarize the recorded evaluator data and render target/torque/speed evidence."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
data = np.load(OUT / "motor_limit_samples.npz")
meta = json.loads((OUT / "experiment.json").read_text())
target = data["processed_target"]
raw = data["raw_action"]
position = data["position_after_rad"]
velocity = data["velocity_after_rad_s"]
computed = data["computed_torque_nm"]
applied = data["applied_torque_estimate_nm"]
effort = data["joint_effort_limits"]
speed = data["joint_vel_limits"]
poslim = data["joint_pos_limits"]
tol = 1e-5
rows = []


def metric(name, values, mask, units):
    row = {"metric": name, "count": int(mask.sum()), "samples": int(mask.size),
           "percent": float(mask.mean() * 100), "max_abs": float(np.max(np.abs(values))), "units": units}
    rows.append(row)
    return row


steer_target_over = (target[:, :, 0] < poslim[:, 0, 0] - tol) | (target[:, :, 0] > poslim[:, 0, 1] + tol)
drive_target_over = np.abs(target[:, :, 1]) > effort[:, 1] + tol
episode_start = np.ones(target.shape[:2], dtype=bool)
episode_start[1:] = data["episode_id"][1:] != data["episode_id"][:-1]
metric("steering_position_target_outside_joint_range", np.rad2deg(target[:, :, 0]), steer_target_over, "deg")
metric("drive_feedforward_target_above_effort_limit", target[:, :, 1], drive_target_over, "N m")
for motor, index in (("steering", 0), ("drive", 1)):
    metric(f"{motor}_raw_action_above_one", raw[:, :, index], np.abs(raw[:, :, index]) > 1.0 + tol, "dimensionless")
    metric(f"{motor}_computed_torque_above_effort_limit", computed[:, :, :, index],
           np.abs(computed[:, :, :, index]) > effort[:, index] + tol, "N m")
    metric(f"{motor}_applied_torque_estimate_above_effort_limit", applied[:, :, :, index],
           np.abs(applied[:, :, :, index]) > effort[:, index] + tol, "N m")
    metric(f"{motor}_speed_above_configured_limit", velocity[:, :, :, index],
           np.abs(velocity[:, :, :, index]) > speed[:, index] + tol, "rad/s")
    metric(f"{motor}_speed_above_limit_by_one_percent", velocity[:, :, :, index],
           np.abs(velocity[:, :, :, index]) > speed[:, index] * 1.01 + tol, "rad/s")

steer_position_over = ((position[:, :, :, 0] < poslim[:, 0, 0] - tol)
                       | (position[:, :, :, 0] > poslim[:, 0, 1] + tol))
metric("steering_measured_position_outside_joint_range", np.rad2deg(position[:, :, :, 0]), steer_position_over, "deg")
valid = data["episode_id"][1:] == data["episode_id"][:-1]
target_rate = np.diff(target[:, :, 0], axis=0) / meta["policy_dt_s"]
metric("steering_target_change_rate_above_speed_limit", target_rate[valid],
       (np.abs(target_rate) > speed[:, 0] + tol)[valid], "rad/s")
tracking_rate = np.abs(target[:, :, 0] - data["position_before_policy_rad"][:, :, 0]) / meta["policy_dt_s"]
metric("steering_rate_needed_to_reach_target_in_one_policy_period", tracking_rate,
       tracking_rate > speed[:, 0] + tol, "rad/s")

prediction = np.empty_like(computed)
prediction[:, :, :, 0] = (data["stiffness_nm_per_rad"][:, None, :, 0]
                         * (target[:, None, :, 0] - data["position_before_rad"][:, :, :, 0])
                         - data["damping_nm_s_per_rad"][:, None, :, 0] * data["velocity_before_rad_s"][:, :, :, 0])
prediction[:, :, :, 1] = (target[:, None, :, 1]
                         - data["damping_nm_s_per_rad"][:, None, :, 1] * data["velocity_before_rad_s"][:, :, :, 1])
law_error = float(np.max(np.abs(prediction - computed)))
clip_error = float(np.max(np.abs(np.clip(computed, -effort, effort) - applied)))
assert law_error < 1e-4, law_error
assert clip_error < 1e-5, clip_error
assert np.max(np.abs(raw * np.array([1.57, 6.28]) - target)) < 1e-5
assert np.all(np.isfinite(computed))

drive_net_within = np.abs(computed[:, 0, :, 1]) <= effort[:, 1] + tol
drive_cancelled = drive_target_over & drive_net_within
drive_conditional = {"excessive_feedforward_targets": int(drive_target_over.sum()),
                     "net_computed_torque_within_limit_first_substep": int(drive_cancelled.sum()),
                     "percent_of_excessive_targets": float(100 * drive_cancelled.sum() / max(1, drive_target_over.sum()))}
if drive_cancelled.any():
    t, env = np.argwhere(drive_cancelled)[0]
    drive_conditional["example"] = {"policy_step": int(t), "environment": int(env),
        "feedforward_target_nm": float(target[t, env, 1]),
        "velocity_rad_s": float(data["velocity_before_rad_s"][t, 0, env, 1]),
        "damping_nm_s_per_rad": float(data["damping_nm_s_per_rad"][t, env, 1]),
        "net_computed_torque_nm": float(computed[t, 0, env, 1])}

steer_env = int(np.argmax(np.max(np.abs(target[:, :, 0]), axis=0)))
drive_env = int(np.argmax(np.max(np.abs(target[:, :, 1]), axis=0)))
summary = {"experiment": meta, "metrics": rows, "drive_excess_but_net_within": drive_conditional,
           "law_reconstruction_max_error_nm": law_error, "estimate_clipping_max_error_nm": clip_error,
           "position_limits_rad": poslim[0].tolist(), "velocity_limits_rad_s": speed[0].tolist(),
           "effort_limits_nm": effort[0].tolist(), "selected_trace_envs": {"steering": steer_env, "drive": drive_env},
           "tolerance": tol,
           "ranges": {name: {"steering": [float(data[name][:, :, 0].min()), float(data[name][:, :, 0].max())],
                              "drive": [float(data[name][:, :, 1].min()), float(data[name][:, :, 1].max())]}
                      for name in ("stiffness_nm_per_rad", "damping_nm_s_per_rad")},
           "physics_effort_target_exceedances": int((np.abs(data["physx_effort_target_nm"][:, :, :, 1]) > effort[:, 1] + tol).sum()),
           "physics_effort_target_equals_policy_drive_max_error_nm": float(np.max(np.abs(data["physx_effort_target_nm"][:, :, :, 1] - target[:, None, :, 1]))),
           "environment_any_steering_target_excess": int(steer_target_over.any(axis=0).sum()),
           "environment_any_drive_target_excess": int(drive_target_over.any(axis=0).sum()),
           "episode_start_samples": int(episode_start.sum()),
           "target_excess_by_episode_phase": {
               "steering": {"at_episode_start": int((steer_target_over & episode_start).sum()),
                            "after_episode_start": int((steer_target_over & ~episode_start).sum())},
               "drive": {"at_episode_start": int((drive_target_over & episode_start).sum()),
                         "after_episode_start": int((drive_target_over & ~episode_start).sum())}},
           "drive_feedforward_range_at_episode_start_nm": [float(target[:, :, 1][episode_start].min()),
                                                          float(target[:, :, 1][episode_start].max())],
           "drive_feedforward_max_abs_after_episode_start_nm": float(np.abs(target[:, :, 1][~episode_start]).max()),
           "torque_statistics": {motor: {"computed_max_abs_nm": float(np.abs(computed[:, :, :, index]).max()),
                "estimated_applied_mean_abs_nm": float(np.abs(applied[:, :, :, index]).mean()),
                "estimated_applied_rms_nm": float(np.sqrt(np.mean(applied[:, :, :, index] ** 2))),
                "estimated_applied_p99_abs_nm": float(np.quantile(np.abs(applied[:, :, :, index]), 0.99))}
                for motor, index in (("steering", 0), ("drive", 1))}}
(OUT / "motor_limit_statistics.json").write_text(json.dumps(summary, indent=2))
with (OUT / "exceedances.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for index, motor in enumerate(("steering", "drive")):
    values = np.rad2deg(target[:, :, 0]).ravel() if index == 0 else target[:, :, 1].ravel()
    lim = 90.0 if index == 0 else float(effort[0, 1])
    ax = axes[0, index]
    ax.hist(values, bins=120, color="#2878a5", log=True)
    ax.axvline(-lim, color="firebrick", ls="--")
    ax.axvline(lim, color="firebrick", ls="--", label=f"configured range +/-{lim:g}")
    ax.set_title(f"{motor}: scaled policy target ({target.shape[0] * target.shape[1]:,} samples)")
    ax.set_xlabel("steering position target (deg)" if index == 0 else "drive feed-forward target (N m)")
    ax.set_ylabel("sample count (log scale)")
    ax.legend(fontsize=8)
    ax = axes[1, index]
    lo = min(float(computed[:, :, :, index].min()), -float(effort[0, index]) * 1.1)
    hi = max(float(computed[:, :, :, index].max()), float(effort[0, index]) * 1.1)
    bins = np.linspace(lo, hi, 160)
    ax.hist(computed[:, :, :, index].ravel(), bins=bins, histtype="step", lw=1.2, log=True, label="computed torque estimate")
    ax.hist(applied[:, :, :, index].ravel(), bins=bins, histtype="step", lw=1.2, log=True, label="clipped applied estimate")
    ax.axvline(-effort[0, index], color="firebrick", ls="--")
    ax.axvline(effort[0, index], color="firebrick", ls="--")
    ax.set_title(f"{motor}: actuator estimates ({computed.shape[0] * computed.shape[1] * computed.shape[2]:,} samples)")
    ax.set_xlabel("torque (N m)")
    ax.set_ylabel("sample count (log scale)")
    ax.legend(fontsize=8)
fig.suptitle("09221945 / model_91850 -- 64 environments, seed 42, full DR\nTorque traces are implicit-actuator estimates, not measured motor torques", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "limit_distributions.png", dpi=160)
plt.close(fig)

fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
policy_time = np.arange(target.shape[0]) * meta["policy_dt_s"]
physics_time = np.arange(target.shape[0] * computed.shape[1]) * meta["physics_dt_s"]
physics_state_time = physics_time + meta["physics_dt_s"]
for index, motor, env in ((0, "steering", steer_env), (1, "drive", drive_env)):
    ax = axes[0, index]
    if index == 0:
        ax.step(policy_time, np.rad2deg(target[:, env, 0]), where="post", lw=0.7, label="requested position")
        ax.plot(physics_state_time, np.rad2deg(position[:, :, env, 0].reshape(-1)), lw=0.7, label="position after physics")
        lim = 90.0
        ax.set_ylabel("position (deg)")
    else:
        ax.step(policy_time, target[:, env, 1], where="post", lw=0.7, label="feed-forward torque target")
        lim = float(effort[env, 1])
        ax.set_ylabel("feed-forward target (N m)")
    ax.axhline(lim, color="firebrick", ls="--", lw=0.8)
    ax.axhline(-lim, color="firebrick", ls="--", lw=0.8)
    ax.set_title(f"{motor}, env {env} (largest absolute target)")
    ax.legend(fontsize=8)
    ax = axes[1, index]
    ax.plot(physics_time, computed[:, :, env, index].reshape(-1), lw=0.6, label="computed torque estimate")
    ax.plot(physics_time, applied[:, :, env, index].reshape(-1), lw=0.6, label="clipped applied estimate")
    ax.axhline(effort[env, index], color="firebrick", ls="--", lw=0.8)
    ax.axhline(-effort[env, index], color="firebrick", ls="--", lw=0.8)
    ax.set_ylabel("estimated torque (N m)")
    ax.legend(fontsize=8)
    ax = axes[2, index]
    ax.plot(physics_state_time, velocity[:, :, env, index].reshape(-1), lw=0.6, color="#377d3c")
    ax.axhline(speed[env, index], color="firebrick", ls="--", lw=0.8)
    ax.axhline(-speed[env, index], color="firebrick", ls="--", lw=0.8)
    ax.set_ylabel("speed (rad/s)")
    ax.set_xlabel("global simulated time (s)")
    for axis in axes[:, index]:
        for step in np.flatnonzero(data["done"][:, env]):
            axis.axvline((step + 1) * meta["policy_dt_s"], color="gray", ls=":", lw=0.5)
        axis.grid(alpha=0.2)
fig.suptitle("Motor targets, estimated torque and simulated speed\nDashed red: configured limits; vertical dotted gray: episode resets", fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "motor_traces.png", dpi=160)
for axis in axes.flat:
    axis.set_xlim(0, 0.6)
fig.suptitle("First 0.6 seconds: requested targets versus estimates and simulated motion\nSame selected environments as motor_traces.png; torque values remain estimates", fontsize=12)
fig.savefig(OUT / "startup_detail.png", dpi=160)
plt.close(fig)

print(json.dumps(summary, indent=2))
