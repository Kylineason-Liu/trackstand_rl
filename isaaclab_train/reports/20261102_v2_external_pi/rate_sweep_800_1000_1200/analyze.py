"""Verify saved PI-rate captures and plot the same 1-rad/s unloaded test."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CASES = {400: HERE.parent / "400_dr", 800: HERE / "800_dr", 1000: HERE / "1000_dr",
         1200: HERE / "1200_dr", 1600: HERE.parent / "1600_dr"}
rows, captures = [], {}
for hz, folder in CASES.items():
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["contract_passed"] and summary["partial_reset_passed"] and summary["timeout_reset_passed"]
    assert summary["domain_randomization"] and summary["num_envs"] == 64 and summary["seed"] == 42
    assert summary["policy_hz"] == 50 and summary["integral_hz"] == 200 and summary["physics_hz"] == hz
    data = np.load(folder / "unloaded.npz")["speed_integral_applied"]
    assert data.shape == (int(hz * .4), 64, 3) and np.isfinite(data).all()
    speed, integral, effort = (data[:, :, index] for index in range(3))
    np.testing.assert_allclose(effort, np.clip(.3 * (1 - speed) + integral, -6.28, 6.28), atol=2e-6, rtol=1e-6)
    for index in range(1, len(data)):
        if index % (hz // 200):
            np.testing.assert_array_equal(integral[index], integral[index - 1])
    late = speed[len(data) // 2 :]
    spans = np.ptp(late, axis=0)
    worst = int(spans.argmax())
    row = dict(
        physics_hz=hz, policy_hz=50, integral_hz=200,
        late_mean_abs_error_rad_s=float(np.abs(late - 1).mean()),
        late_max_span_rad_s=float(spans.max()), late_median_span_rad_s=float(np.median(spans)),
        # Descriptive comparison threshold only, never an experiment abort or physical limit.
        envs_late_span_above_0p1_rad_s=int((spans > .1).sum()),
        max_applied_torque_nm=float(np.abs(effort).max()), worst_env=worst,
    )
    np.testing.assert_allclose(row["late_max_span_rad_s"], summary["unloaded_late_speed_peak_to_peak_max_rad_s"])
    np.testing.assert_allclose(row["late_mean_abs_error_rad_s"], summary["unloaded_late_mean_abs_speed_error_rad_s"])
    rows.append(row)
    captures[hz] = data, worst
    with (HERE / f"{hz}_dr_worst_env_trace.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time_s", "target_rad_s", "speed_rad_s", "integral_nm", "applied_torque_nm"))
        writer.writerows((index / hz, 1., *sample) for index, sample in enumerate(data[:, worst, :]))

nominal_folder = HERE / "1200_nominal"
nominal = json.loads((nominal_folder / "summary.json").read_text())
assert nominal["contract_passed"] and not nominal["domain_randomization"]
nominal_data = np.load(nominal_folder / "unloaded.npz")["speed_integral_applied"]
np.testing.assert_allclose(
    nominal_data[:, :, 2], np.clip(.3 * (1 - nominal_data[:, :, 0]) + nominal_data[:, :, 1], -6.28, 6.28),
    atol=2e-6, rtol=1e-6,
)
for index in range(1, len(nominal_data)):
    if index % 6:
        np.testing.assert_array_equal(nominal_data[index, :, 1], nominal_data[index - 1, :, 1])

with (HERE / "comparison.csv").open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(HERE / "comparison.json").write_text(json.dumps({
    "domain_randomization": True, "num_envs": 64, "seed": 42,
    "target_rad_s": 1., "duration_s": .4, "late_window_s": [.2, .4],
    "fixture": "Whole bicycle free-falling from root z=3 m, clear of ground; no fixed frame.",
    "p_nm_s_per_rad": .3, "i_nm_per_rad": 3.3, "cases": rows,
    "nominal_1200": nominal,
    "note": "800/1000/1200 are new; 400/1600 reused unchanged. No trained policy or hardware experiment.",
}, indent=2) + "\n")

fig, axes = plt.subplots(3, 4, figsize=(17, 9), sharex=True, constrained_layout=True)
for column, hz in enumerate((800, 1000, 1200, 1600)):
    data, worst = captures[hz]
    time = np.arange(len(data)) / hz
    for channel, label in enumerate(("Rear speed (rad/s)", "Integral torque (Nm)", "Applied torque (Nm)")):
        ax = axes[channel, column]
        ax.fill_between(time, data[:, :, channel].min(axis=1), data[:, :, channel].max(axis=1),
                        color="0.85", label="64-env range")
        ax.plot(time, data[:, worst, channel], color="#1764aa", linewidth=1., label=f"Env {worst}")
        if channel == 0:
            ax.axhline(1., color="#bb2222", linestyle="--", linewidth=1., label="Target 1 rad/s")
            ax.set_title(f"{hz} Hz physics / P loop")
        if column == 0:
            ax.set_ylabel(label)
        if channel == 2:
            ax.set_xlabel("Simulated time (s)")
        ax.axvspan(.2, .4, color="#f7c75b", alpha=.08)
        ax.grid(alpha=.2)
        ax.legend(loc="lower right", fontsize=7)
        ax.set_xlim(0, .4)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
fig.suptitle("Same constant rear target | 64 DR environments, seed 42 | target 50 Hz, integral 200 Hz\n"
             "Whole bike airborne/free; each column highlights its worst late-span environment; y scales differ")
for extension in ("png", "svg", "pdf"):
    fig.savefig(HERE / f"pi_rate_comparison.{extension}", dpi=160)
plt.close(fig)

baseline = json.loads((HERE / "preservation_before.json").read_text())
root = HERE.parents[3]
for name, expected in baseline["files"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
(HERE / "verification.json").write_text(json.dumps({
    "passed": True, "independent_npz_p_plus_i_clamp_and_integral_hold_checks": True,
    "existing_files_unchanged": len(baseline["files"]),
    "grounded_random_steps_per_new_case": 100, "new_dr_cases": [800, 1000, 1200],
    "new_nominal_cases": [1200], "policy_or_hardware_evaluation": False,
}, indent=2) + "\n")
print(json.dumps(rows, indent=2))
