"""Additional source-timestamp, wire-contract and timing checks for analyze.py."""

steer_sent = col("motor_control", "steering_send_success") > .5
rear_sent = col("motor_control", "drive_send_success") > .5
wire = {
    "successful_recorded_steering_sends": int(steer_sent.sum()),
    "successful_recorded_rear_sends": int(rear_sent.sum()),
    "steering_encoding_max_abs_error_rad": np.max(abs(
        col("motor_control", "mit_encoded_position_rad")[steer_sent]
        - col("motor_control", "requested_steering_target_rad")[steer_sent])),
    "rear_encoding_max_abs_error_nm": np.max(abs(
        col("motor_control", "drive_encoded_torque_nm")[rear_sent]
        - col("motor_control", "requested_drive_target_nm")[rear_sent])),
    "encoded_gains_and_zero_quantization": {
        name: np.unique(col("motor_control", name)[steer_sent]) for name in (
            "mit_encoded_kp_nm_per_motor_rad", "mit_encoded_kd_nm_s_per_motor_rad",
            "mit_encoded_velocity_motor_rad_s", "mit_encoded_feedforward_torque_motor_nm")
    },
}
true_mit_intervals = []
for ep in EPISODES:
    # This field is populated at each actual send; discard each enable's first
    # value because it legitimately spans the preceding disabled interval.
    true_mit_intervals.extend(col("motor_control", "mit_actual_send_interval_ms")[ep["_sm"]][1:])
timing["motor_reported_actual_mit_send_interval_ms"] = stats(true_mit_intervals)
assert wire["steering_encoding_max_abs_error_rad"] <= 3 / 65535 + 1e-12
assert wire["rear_encoding_max_abs_error_nm"] <= .000151
assert np.nanmax(abs(col("motor_control", "mit_native_target_rad")[steer_sent]
                     - (col("motor_control", "mit_native_zero_rad")[steer_sent]
                        - col("motor_control", "requested_steering_target_rad")[steer_sent]))) < 1e-12
assert np.max(abs(col("motor_control", "drive_encoded_torque_nm")[rear_sent]
                  + .30000001192092896 * col("motor_control", "drive_encoded_current_a")[rear_sent])) < 1e-12

# No fitted shift is applied to presentation traces or headline tracking RMS.
# This estimates aggregate apparent alignment, NOT a motor-internal delay.
lag_results = {}
for channel, field, feedback_index, mask_key in (
    ("steering", "mit_encoded_position_rad", 1, "_sm"),
    ("rear_current_torque", "drive_encoded_torque_nm", 7, "_dm"),
):
    scores, counts = [], []
    for delay_ms in range(101):
        errors = []
        for ep in EPISODES:
            sm = ep[mask_key]
            ss = ep["_ss"] & (ST[:, 0] > ep["start_s"] + .1) & (ST[:, 0] < ep["stop_s"] - .01)
            target = held(MC[sm, 0], col("motor_control", field)[sm], ST[ss, 0] - delay_ms / 1000)
            err = ST[ss, feedback_index] - target
            errors.extend(err[np.isfinite(err)])
        scores.append(rms(errors))
        counts.append(len(errors))
    best = int(np.nanargmin(scores))
    lag_results[channel] = dict(best_offset_ms=best, rms_without_offset=scores[0], rms_at_best_offset=scores[best],
                                valid_samples_without_offset=counts[0], valid_samples_at_best_offset=counts[best],
                                unit="rad" if channel == "steering" else "Nm",
                                interpretation="1 ms grid, 0..100 ms; excludes first 100 ms and final 10 ms. "
                                "Includes dynamics, feedback age and publication timing; not identified transport delay.")
    save_csv(f"{channel}_alignment_scan.csv", [dict(offset_ms=i, rms=scores[i], samples=counts[i])
                                               for i in range(101)])

active_st = np.zeros(len(ST), dtype=bool)
for ep in EPISODES:
    active_st |= ep["_ss"]
motion = {
    "active_state_samples": int(active_st.sum()),
    "steering_speed_abs_stats_rad_s": stats(abs(ST[active_st, 3])),
    "rear_speed_abs_stats_rad_s": stats(abs(ST[active_st, 4])),
    "steering_speed_above_90pct_of_nominal_limit_fraction": np.mean(abs(ST[active_st, 3]) >= .9 * 23.9),
    "rear_speed_above_90pct_of_nominal_limit_fraction": np.mean(abs(ST[active_st, 4]) >= .9 * 61.6),
    "steering_current_torque_abs_stats_nm": stats(abs(ST[active_st, 6])),
    "rear_current_torque_abs_stats_nm": stats(abs(ST[active_st, 7])),
    "steering_current_torque_above_sim_cap_sample_count": np.sum(abs(ST[active_st, 6]) > 5.6),
    "rear_current_torque_above_sim_cap_sample_count": np.sum(abs(ST[active_st, 7]) > 6.28),
    "initial_roll_deg_stats": stats([ep["initial_roll_deg"] for ep in EPISODES]),
    "initial_steering_rad_stats": stats([ep["initial_steering_rad"] for ep in EPISODES]),
    "initial_drive_phase_observation_max_abs_rad": max(abs(OBS[ep["_oi"][0], 7]) for ep in EPISODES),
}

# Per-episode policy and packet rates avoid dividing across idle stretches.
for ep, public in zip(EPISODES, public_episodes):
    oi = ep["_oi"]
    mono = col("policy_timing", "monotonic_time_s")[SUCCESS[oi]]
    public["effective_inference_rate_hz"] = (len(mono) - 1) / np.ptp(mono) if len(mono) > 1 else np.nan
    for key, prefix in (("_sm", "mit"), ("_dm", "rear_c0")):
        mono = col("motor_control", "monotonic_time_s")[ep[key]]
        public[f"recorded_effective_{prefix}_rate_hz"] = (len(mono) - 1) / np.ptp(mono) if len(mono) > 1 else np.nan

summary.update(wire_contract_checks=wire, apparent_alignment_scan=lag_results, measured_motion=motion,
               rate_caveat="Recorded successful motor sends may omit ticks if telemetry is not published/captured. "
                           "Nominal configuration is 200 Hz; source code schedules the next packet after a send.")
summary["contract_validation"].update(wire)
summary["episodes"] = public_episodes
rear_response = []
rear_conditioned = []
for offset_ms in (0, 5, 10, 15):
    paired = []
    for ep in EPISODES:
        sm = ep["_dm"]
        ss = ep["_ss"] & (ST[:, 0] > ep["start_s"] + .1) & (ST[:, 0] < ep["stop_s"] - .01)
        values = ST[ss]
        target = held(MC[sm, 0], col("motor_control", "drive_encoded_torque_nm")[sm],
                      values[:, 0] - offset_ms / 1000)
        for row, request in zip(values, target):
            if np.isfinite(request) and np.isfinite(row[7]):
                paired.append((request, row[7], row[4]))
                if offset_ms == 0:
                    rear_response.append(dict(episode=ep["episode"], feedback_bag_s=row[0],
                                              held_encoded_request_nm=request, a1_current_torque_nm=row[7],
                                              rear_speed_rad_s=row[4]))
    paired = np.asarray(paired)
    requested, measured, speed = paired.T
    groups = {
        "all": np.ones(len(paired), dtype=bool),
        "small_request_abs_le_1nm": abs(requested) <= 1,
        "large_request_abs_gt_3nm": abs(requested) > 3,
        "large_request_high_speed_motoring": (abs(requested) > 3) & (abs(speed) > 40) & (requested * speed > 0),
        "large_request_lower_speed_motoring": (abs(requested) > 3) & (abs(speed) <= 40) & (requested * speed > 0),
        "large_request_braking": (abs(requested) > 3) & (requested * speed < 0),
    }
    for name, mask in groups.items():
        rear_conditioned.append(dict(offset_ms=offset_ms, group=name, n=int(mask.sum()),
                                     mean_abs_requested_nm=np.mean(abs(requested[mask])),
                                     mean_abs_current_torque_nm=np.mean(abs(measured[mask])),
                                     torque_error_rms_nm=rms(requested[mask] - measured[mask])))
summary["rear_response_conditioned"] = rear_conditioned
save_csv("rear_response_pairs.csv", rear_response)
save_csv("rear_response_conditioned.csv", rear_conditioned)
save_csv("episodes.csv", public_episodes)
save_json("contract_checks.json", summary["contract_validation"])
save_json("statistics.json", summary)
print("EXTRA", json.dumps(clean({"wire": wire, "alignment": lag_results, "motion": motion}), indent=2))
