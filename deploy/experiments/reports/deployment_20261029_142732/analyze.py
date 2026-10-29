"""Offline analysis of the October 29 deployment; never opens ROS or CAN.

Run from any directory: python3 <this-file>. The CRC-checked MCAP decode is
archived beside this script. Uses NumPy, Matplotlib and ONNX Runtime.
All cross-topic figures use MCAP reception times; rate/duration calculations
use the recorded source monotonic clocks where available.
"""

from __future__ import annotations

from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
A = dict(np.load(HERE / "decoded_data.npz"))
with gzip.open(HERE / "decoded_details.json.gz", "rt") as stream:
    D = json.load(stream)
SCHEMAS = D["schemas"]
MC, ST, PT, OBS, RAW, IMU, CMD = (A[k] for k in (
    "motor_control", "motor_state", "policy_timing", "observation", "raw_action", "imu", "motor_command"
))


def col(key, name):
    return A[key][:, 1 + SCHEMAS[key].index(name)]


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def save_json(name, value):
    (HERE / name).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def save_csv(name, rows):
    if rows:
        with (HERE / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def held(t, v, q, until=np.inf):
    """Causal zero-order hold; no future extrapolation or disabled tail."""
    idx = np.searchsorted(t, q, side="right") - 1
    out = np.full(np.shape(q), np.nan)
    valid = (idx >= 0) & (q < until)
    out[valid] = v[idx[valid]]
    return out


def stats(values):
    x = np.asarray(values)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return dict(zip(("n", "min", "median", "p95", "p99", "max"),
                    (len(x), *np.percentile(x, [0, 50, 95, 99, 100]))))


def rms(values):
    x = np.asarray(values)
    x = x[np.isfinite(x)]
    return np.sqrt(np.mean(x * x)) if len(x) else np.nan


def finite_percentile(values, p):
    x = np.asarray(values)
    x = x[np.isfinite(x)]
    return np.percentile(x, p) if len(x) else np.nan


def branch(q):
    return 2 * np.arctan2(np.sin(q / 2), np.cos(q / 2))


def gravity(q):
    """Independent quaternion rotation-matrix calculation, xyzw input."""
    q = q / np.linalg.norm(q, axis=1)[:, None]
    x, y, z, w = q.T
    return np.column_stack((2 * (x * z - w * y), 2 * (y * z + w * x),
                            1 - 2 * (x * x + y * y))) * -1


G = gravity(IMU[:, 2:6])
ROLL = np.rad2deg(np.arctan2(-G[:, 1], -G[:, 2]))
SUCCESS = np.flatnonzero(col("policy_timing", "inference_success") > 0.5)
ENABLED = CMD[CMD[:, 1] > 0.5]
assert len(SUCCESS) == len(OBS) == len(RAW) == len(ENABLED) == 1101
assert np.max(abs(OBS[:, 0] - RAW[:, 0])) < 0.005
assert np.max(abs(OBS[:, 0] - ENABLED[:, 0])) < 0.01

STARTS = [x for x in D["text"]["policy_event"] if x["mode"] == "balancing"]
EPISODES = []
OBS_EP = np.zeros(len(OBS), dtype=int)
SLEWED = np.empty(len(OBS))
RECONSTRUCTED_IMU = np.empty((len(OBS), 5))
RECONSTRUCTED_MOTOR = np.empty((len(OBS), 6))
IMU_INPUT_INDEX = np.empty(len(OBS), dtype=int)
MOTOR_INPUT_INDEX = np.empty(len(OBS), dtype=int)
IMU_MATCH_ERRORS, MOTOR_MATCH_ERRORS = [], []

for start in STARTS:
    end = next(x for x in D["text"]["policy_event"] if x["episode"] == start["episode"]
               and x["sequence"] >= start["sequence"] and x["mode"] != "balancing")
    seq = col("policy_timing", "sequence")[SUCCESS]
    oi = np.flatnonzero((seq >= start["sequence"]) & (seq < end["sequence"]))
    OBS_EP[oi] = start["episode"]
    b, e = start["t"], end["t"]
    cp = (MC[:, 0] >= b - 0.015) & (MC[:, 0] < e)
    sm = cp & (col("motor_control", "steering_send_success") > 0.5)
    dm = cp & (col("motor_control", "drive_send_success") > 0.5)
    # Disable acknowledgements can arrive before the policy observes the fault.
    active_end = next((x["t"] for x in D["text"]["motor_event"]
                       if x["t"] > b and not x["outputs_active"]), e)
    active_end = min(active_end, e)
    ss = (ST[:, 0] >= b) & (ST[:, 0] < active_end)
    sm &= MC[:, 0] < active_end
    dm &= MC[:, 0] < active_end
    s_target = held(MC[sm, 0], col("motor_control", "mit_encoded_position_rad")[sm], ST[ss, 0], active_end)
    d_target = held(MC[dm, 0], col("motor_control", "drive_encoded_torque_nm")[dm], ST[ss, 0], active_end)
    err_deg = np.rad2deg(ST[ss, 1] - s_target)
    d_err = ST[ss, 7] - d_target
    later_imu = (IMU[:, 0] >= b) & (IMU[:, 0] <= e + 0.04)
    reasons = end["latched_fault_reasons"]
    reason = ",".join(reasons) if reasons else "RC_down"
    duration = end["monotonic_time_s"] - start["monotonic_time_s"]
    near = lambda j: np.mean(abs(RAW[oi, j]) >= 0.95)
    record = dict(episode=start["episode"], start_s=b, stop_s=e, duration_s=duration,
                  n_actions=len(oi), reason=reason, rear_zero_rad=start["rear_episode_zero_rad"],
                  initial_roll_deg=held(IMU[:, 0], ROLL, np.array([b]))[0],
                  max_abs_roll_through_stop_plus_40ms_deg=np.max(abs(ROLL[later_imu])),
                  initial_steering_rad=OBS[oi[0], 6],
                  steering_tracking_rms_deg=rms(err_deg),
                  steering_tracking_p95_abs_deg=finite_percentile(abs(err_deg), 95),
                  steering_tracking_peak_abs_deg=finite_percentile(abs(err_deg), 100),
                  rear_current_torque_tracking_rms_nm=rms(d_err),
                  peak_steering_speed_rad_s=np.max(abs(ST[ss, 3])) if ss.any() else np.nan,
                  peak_rear_speed_rad_s=np.max(abs(ST[ss, 4])) if ss.any() else np.nan,
                  peak_abs_rear_current_torque_nm=np.nanmax(abs(ST[ss, 7])) if ss.any() else np.nan,
                  steering_near_endpoint_fraction=near(1), rear_near_endpoint_fraction=near(2),
                  successful_recorded_mit_sends=int(sm.sum()), successful_recorded_c0_sends=int(dm.sum()))
    previous = np.clip(OBS[oi[0], 6], -1.57, 1.57)
    for i in oi:
        # The initial limiter state uses the same encoder snapshot as first obs.
        previous += np.clip(np.clip(RAW[i, 1] * 1.57, -1.57, 1.57) - previous, -0.478, 0.478)
        SLEWED[i] = previous
        q = OBS[i, 0]
        indices = np.flatnonzero(abs(IMU[:, 0] - q) < 0.06)
        candidates = np.column_stack((G[indices, 1:3], IMU[indices, 6:9]))
        errors = np.max(abs(candidates - OBS[i, 1:6]), axis=1)
        j = np.argmin(errors)
        IMU_INPUT_INDEX[i] = indices[j]
        RECONSTRUCTED_IMU[i] = candidates[j]
        IMU_MATCH_ERRORS.append(errors[j])
        indices = np.flatnonzero(abs(ST[:, 0] - q) < 0.06)
        candidates = np.column_stack((ST[indices, 1], branch(ST[indices, 2] - record["rear_zero_rad"]),
                                       ST[indices, 3:5], np.sin(ST[indices, 1]), np.cos(ST[indices, 1])))
        errors = np.max(abs(candidates - OBS[i, 6:]), axis=1)
        j = np.argmin(errors)
        MOTOR_INPUT_INDEX[i] = indices[j]
        RECONSTRUCTED_MOTOR[i] = candidates[j]
        MOTOR_MATCH_ERRORS.append(errors[j])
    record["steering_slew_active_fraction"] = np.mean(abs(SLEWED[oi] - RAW[oi, 1] * 1.57) > 1e-6)
    record["_oi"], record["_end"], record["_sm"], record["_dm"], record["_ss"] = oi, end, sm, dm, ss
    record["_steer_error"], record["_rear_error"] = err_deg, d_err
    EPISODES.append(record)

assert np.all(OBS_EP > 0)
assert np.max(IMU_MATCH_ERRORS) < 3e-7, max(IMU_MATCH_ERRORS)
assert np.max(MOTOR_MATCH_ERRORS) < 3e-6, max(MOTOR_MATCH_ERRORS)
assert np.all(IMU[IMU_INPUT_INDEX, 0] <= OBS[:, 0])
assert np.all(ST[MOTOR_INPUT_INDEX, 0] <= OBS[:, 0])
STEER_FLOW_ERROR = abs(SLEWED - ENABLED[:, 2])
DRIVE_FLOW_ERROR = abs(RAW[:, 2] * 6.28 - ENABLED[:, 3])
assert np.max(STEER_FLOW_ERROR) < 1e-7, max(STEER_FLOW_ERROR)
assert np.max(DRIVE_FLOW_ERROR) < 1e-7, max(DRIVE_FLOW_ERROR)

model = ROOT / "deploy/source/trackstand_deploy/models/trackstand_rl_rsl_2026-10-21_11-18-29_model_2500.onnx"
options = ort.SessionOptions()
options.intra_op_num_threads = options.inter_op_num_threads = 1
session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
prediction = np.array([session.run(None, {"observation": row[1:].astype(np.float32)[None]})[0][0] for row in OBS])
assert np.array_equal(prediction, RAW[:, 1:])
assert len({s["model_sha256"] for s in D["text"]["status"]}) == 1
assert hashlib.sha256(model.read_bytes()).hexdigest() == D["text"]["status"][0]["model_sha256"]

active_mc = np.zeros(len(MC), dtype=bool)
mit_intervals, drive_intervals = [], []
for ep in EPISODES:
    active_mc |= ((MC[:, 0] >= ep["start_s"]) & (MC[:, 0] < ep["stop_s"]))
    for mask, collection in ((ep["_sm"], mit_intervals), (ep["_dm"], drive_intervals)):
        mono = col("motor_control", "monotonic_time_s")[mask]
        collection.extend(np.diff(mono) * 1000)
active_mc &= col("motor_control", "outputs_active") > 0.5
active_pt = SUCCESS
timing = {"policy": {name: stats(col("policy_timing", name)[active_pt]) for name in (
    "cycle_interval_ms", "work_duration_ms", "inference_ms", "imu_age_ms", "motor_age_ms", "rc_age_ms")},
          "motor": {name: stats(col("motor_control", name)[active_mc]) for name in (
              "cycle_interval_ms", "work_duration_ms", "command_age_ms", "steering_pair_age_ms",
              "drive_pair_age_ms", "steering_driver_status_age_ms", "drive_driver_status_age_ms")},
          "recorded_successful_mit_interval_ms": stats(mit_intervals),
          "recorded_successful_c0_interval_ms": stats(drive_intervals)}

faults = []
for ep in EPISODES:
    if ep["reason"] not in ("motor_fault", "motor_state_timeout"):
        continue
    events = [x for x in D["text"]["motor_event"] if ep["start_s"] < x["t"] < ep["stop_s"] + 0.06]
    latch = next(x for x in events if x["fault_latched"])
    i = np.argmin(abs(col("motor_control", "monotonic_time_s") - latch["monotonic_time_s"]))
    names = ("steering_pair_age_ms", "drive_pair_age_ms", "steering_driver_status_age_ms",
             "drive_driver_status_age_ms", "command_age_ms")
    faults.append(dict(episode=ep["episode"], event=latch,
                       nearest_control_telemetry_delta_ms=1000 * (MC[i, 0] - latch["t"]),
                       nearest_control_ages={name: col("motor_control", name)[i] for name in names}))

all_steer_err = np.concatenate([ep["_steer_error"] for ep in EPISODES])
all_rear_err = np.concatenate([ep["_rear_error"] for ep in EPISODES])
public_episodes = [{k: v for k, v in ep.items() if not k.startswith("_")} for ep in EPISODES]
summary = {
    "bag": {key: D[key] for key in ("bag", "bytes", "sha256", "duration_s", "counts", "issues")},
    "model": D["text"]["status"][0],
    "user_fixture": "Both wheels on ground, bike unsupported, steering straight at I/O restart; no setting changes.",
    "episodes": public_episodes,
    "stop_reason_counts": Counter(ep["reason"] for ep in EPISODES),
    "all_policy_interval_duration_s": stats([ep["duration_s"] for ep in EPISODES]),
    "roll_stop_duration_s": stats([ep["duration_s"] for ep in EPISODES if ep["reason"] == "roll_or_inversion_safety"]),
    "timing": timing, "feedback_stops": faults,
    "contract_validation": {
        "paired_observations_actions_enabled_commands": len(OBS),
        "independent_imu_reconstruction_max_abs_error": np.max(IMU_MATCH_ERRORS),
        "independent_encoder_reconstruction_max_abs_error": np.max(MOTOR_MATCH_ERRORS),
        "onnx_saved_observations_max_abs_action_error": np.max(abs(prediction - RAW[:, 1:])),
        "steering_clip_and_50hz_slew_command_max_abs_error_rad": np.max(STEER_FLOW_ERROR),
        "rear_scale_command_max_abs_error_nm": np.max(DRIVE_FLOW_ERROR),
        "raw_action_min": RAW[:, 1:].min(axis=0), "raw_action_max": RAW[:, 1:].max(axis=0),
        "motor_state_nonfinite_torque_samples": np.sum(~np.isfinite(ST[:, 6:8]), axis=0),
        "steering_slew_active_fraction": np.mean(abs(SLEWED - RAW[:, 1] * 1.57) > 1e-6),
        "backend_steering_clip_ticks": np.sum(col("motor_control", "steering_target_clipped")[active_mc] > 0.5),
        "backend_rear_clip_ticks": np.sum(col("motor_control", "drive_target_clipped")[active_mc] > 0.5),
        "backend_rear_damping_max_abs_nm": np.nanmax(abs(col("motor_control", "drive_damping_torque_nm")[active_mc])),
    },
    "reported_unshifted_steering_tracking_rms_deg": rms(all_steer_err),
    "reported_unshifted_rear_current_torque_tracking_rms_nm": rms(all_rear_err),
    "steering_near_raw_endpoint_fraction": np.mean(abs(RAW[:, 1]) >= .95),
    "rear_near_raw_endpoint_fraction": np.mean(abs(RAW[:, 2]) >= .95),
    "imu_quaternion_norm": stats(np.linalg.norm(IMU[:, 2:6], axis=1)),
    "whole_bag_message_rates_hz": {
        k: (len(A[k]) - 1) / np.ptp(A[k][:, 0]) for k in ("kickstands_up", "motor_command", "policy_timing", "imu")
    },
}
save_csv("episodes.csv", public_episodes)
save_json("statistics.json", summary)
save_json("contract_checks.json", summary["contract_validation"])
print(json.dumps(clean({k: v for k, v in summary.items() if k not in ("bag", "episodes", "model")}), indent=2))

# Every diagnostic policy input/action/command pairing is retained at native times.
rows = []
for i in range(len(OBS)):
    row = dict(episode=OBS_EP[i], observation_bag_s=OBS[i, 0], action_bag_s=RAW[i, 0],
               command_bag_s=ENABLED[i, 0], raw_steering=RAW[i, 1], raw_rear=RAW[i, 2],
               scaled_steering_rad=RAW[i, 1] * 1.57, scaled_rear_nm=RAW[i, 2] * 6.28,
               command_steering_rad=ENABLED[i, 2], command_rear_nm=ENABLED[i, 3])
    row.update(zip(SCHEMAS["observation"], OBS[i, 1:]))
    rows.append(row)
save_csv("policy_samples.csv", rows)
np.savez_compressed(HERE / "validated_analysis.npz", episode=OBS_EP, reconstructed_imu=RECONSTRUCTED_IMU,
                    reconstructed_motor=RECONSTRUCTED_MOTOR, model_action=prediction,
                    imu_input_index=IMU_INPUT_INDEX, motor_input_index=MOTOR_INPUT_INDEX,
                    steering_slewed_rad=SLEWED, enabled_motor_command=ENABLED)

# Plotting and optional saved-simulation context are in a report-local module.
exec(compile((HERE / "extra_metrics.py").read_text(), str(HERE / "extra_metrics.py"), "exec"))
exec(compile((HERE / "plots.py").read_text(), str(HERE / "plots.py"), "exec"))
