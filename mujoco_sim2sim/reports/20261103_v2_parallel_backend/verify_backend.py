"""Verify measured v2 command flow, episode sampling and old-file preservation."""

import hashlib
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_run(directory):
    meta_path = next(directory.glob("*_metadata.json"))
    meta = json.loads(meta_path.read_text())
    base = str(meta_path).removesuffix("_metadata.json")
    trace = np.load(base + "_evaluation.npz")
    mask = trace["active_mask"]
    steps, num_envs = mask.shape
    assert meta["completed_policy_steps"] == steps
    assert meta["environment_transitions"] == int(mask.sum())
    assert meta["completed"]
    assert meta["policy_hz"] == 50 and meta["integral_hz"] == 200 and meta["physics_hz"] == 1200
    assert trace["physics_motor_torque_nm"].shape == (steps, num_envs, 24, 2)
    assert trace["observation"].shape == (steps, num_envs, 11)
    assert np.isfinite(trace["observation"][mask]).all()
    assert np.isfinite(trace["reward"][mask]).all()
    action = trace["raw_action"]
    assert np.isfinite(action[mask]).all() and (np.abs(action[mask]) <= 1).all()
    target = np.clip(action[:, :, 1] * np.float32(61.6), -61.6, 61.6)
    target_error = np.max(np.abs(target[mask] - trace["held_drive_target_rad_s"][mask]))
    assert target_error == 0
    error = trace["physics_drive_error_rad_s"]
    measured_error = trace["held_drive_target_rad_s"][:, :, None] - trace["physics_drive_velocity_rad_s"]
    assert np.array_equal(error[mask], measured_error[mask])
    computed = trace["physics_computed_torque_nm"]
    integral = trace["physics_drive_integral_nm"]
    p_i_error = np.max(np.abs(computed[:, :, :, 1][mask] - (.30 * error + integral)[mask]))
    assert p_i_error == 0
    expected_torque = np.clip(computed, [-5.6, -6.28], [5.6, 6.28])
    torque_error = np.max(np.abs(trace["physics_motor_torque_nm"][mask] - expected_torque[mask]))
    assert torque_error < 1e-12
    reference_i, reference_integral, previous = (np.zeros(num_envs) for _ in range(3))
    initialized = np.zeros(num_envs, dtype=bool)
    steering = np.zeros(num_envs)
    resets = trace["reset_events"]
    i_max_error, slew_max_error = 0., 0.
    for t in range(steps):
        ids = np.flatnonzero(mask[t])
        desired = np.clip(action[t, ids, 0] * np.float32(1.57), -1.57, 1.57)
        steering[ids] += np.clip(desired - steering[ids], -.478, .478)
        slew_max_error = max(slew_max_error, float(np.max(np.abs(
            steering[ids] - trace["held_steering_target_rad"][t, ids]))))
        for s in range(0, 24, 6):
            e = error[t, ids, s]
            increment = np.where(initialized[ids], .5 * (e + previous[ids]) * .005, 0.)
            proposed = 3.3 * (reference_integral[ids] + increment)
            candidate = np.clip(proposed, -6.28, 6.28)
            delta = candidate - reference_i[ids]
            total = .3 * e + candidate
            frozen = ((total > 6.28) & (delta > 0)) | ((total < -6.28) & (delta < 0))
            reference_i[ids] = np.where(frozen, reference_i[ids], candidate)
            reference_integral[ids] = reference_i[ids] / 3.3
            previous[ids], initialized[ids] = e, True
            i_max_error = max(i_max_error, float(np.max(np.abs(
                integral[t, ids, s:s+6] - reference_i[ids, None]))))
        ended = resets[resets[:, 0] == t + 1, 1].astype(int)
        reference_i[ended], reference_integral[ended], previous[ended] = 0., 0., 0.
        initialized[ended], steering[ended] = False, 0.
    assert i_max_error < 1e-12 and slew_max_error < 1e-12
    summary = json.loads(Path(base + "_summary.json").read_text())
    records = meta["episodes"]
    assert summary["completed_episodes"] == len(records)
    assert sum(e["length"] for e in records) == meta["environment_transitions"]
    assert meta["completed_per_env"] == meta["episode_quotas"]
    counts = {term: sum(e["terminations"][term] for e in records) for term in meta["termination_names"]}
    assert summary["termination_counts"] == counts
    histogram = np.loadtxt(base + "_epi_len_histogram.csv", delimiter=",", skiprows=1)
    assert int(histogram[:, 2].sum()) == len(records)
    return dict(metadata=str(meta_path.relative_to(ROOT)), completed_episodes=len(records),
                policy_steps=steps, environment_transitions=int(mask.sum()),
                physics_steps=int(mask.sum()) * 24, integral_ticks=int(mask.sum()) * 4,
                mean_episode_seconds=summary["mean_episode_length_s"],
                termination_counts=counts, mean_roll_rms_deg=summary["roll_rms_deg"],
                maximum_target_scale_error=target_error, maximum_P_plus_I_error=p_i_error,
                maximum_submitted_torque_error=torque_error, maximum_integral_reconstruction_error=i_max_error,
                maximum_steering_slew_error=slew_max_error,
                max_motor_torque_nm=np.max(np.abs(trace["physics_motor_torque_nm"][mask]), axis=(0, 1)).tolist(),
                max_absolute_integral_nm=float(np.max(np.abs(integral[mask]))))


runs = {name: check_run(OUT / "validation" / name) for name in ("no_dr", "dr_smoke")}
before = json.loads((OUT / "preservation_before.json").read_text())
after = {}
for name, info in before.items():
    path = ROOT / name
    assert path.is_file(), name
    after[name] = dict(bytes=path.stat().st_size, sha256=digest(path))
    assert after[name] == info, f"Pre-existing file changed: {name}"
(OUT / "preservation_after.json").write_text(json.dumps(after, indent=2) + "\n")
new_source = list((ROOT / "mujoco_sim2sim/src/newbike_mujoco_v2").glob("*.py"))
new_source += [ROOT / f"mujoco_sim2sim/scripts/run_deploy_{kind}_v2.py" for kind in ("eval", "viewer")]
new_source += list((ROOT / "mujoco_sim2sim/tests").glob("test_v2*.py"))
assert all("from newbike_mujoco " not in p.read_text() and "import newbike_mujoco\n" not in p.read_text()
           for p in new_source)
result = dict(passed=True, runs=runs, original_mujoco_files_unchanged=len(before),
              original_bytes_preserved=sum(i["bytes"] for i in before.values()),
              new_production_and_test_sha256={str(p.relative_to(ROOT)): digest(p) for p in sorted(new_source)},
              gpu_used=False, hardware_used=False)
(OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
