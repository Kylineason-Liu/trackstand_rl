"""Recover episode DR values without attributing an unmatched repeat to old data."""

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def roll_xyzw(quaternion):
    x, y, z, w = quaternion
    return float(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))


def main():
    repeat = HERE / "dr_metadata/isaac_repeat"
    original = HERE / "isaac400/motor_samples.npz"
    repeated = repeat / "motor_samples.npz"
    checks = {}
    with np.load(original) as old, np.load(repeated) as new:
        assert old.files == new.files
        for key in old.files:
            a, b = old[key], new[key]
            checks[key] = a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    assert all(checks.values()), f"Repeat differs: {[key for key, ok in checks.items() if not ok]}"
    output = dict(
        meaning="Actual per-environment, per-episode draws; full precision. Figures round to six significant digits.",
        isaac_recovery=dict(all_original_arrays_byte_identical=True, array_checks=checks,
                            original_sha256=digest(original), repeat_sha256=digest(repeated),
                            snapshot_sha256=digest(repeat / "dr_draws.json")),
        runs={})
    for tag in ("isaac400", "mujoco400", "mujoco800"):
        folder = HERE / tag
        native = folder / "motor_samples.npz" if tag == "isaac400" else next(folder.glob("*_evaluation.npz"))
        with np.load(native) as data:
            done = data["done"] if tag == "isaac400" else np.any(data["termination_flags"], axis=-1)
        source = repeat / "dr_draws.json" if tag == "isaac400" else folder / "dr_draws.json"
        draws = json.loads(source.read_text())
        episodes = []
        for draw in draws:
            env, number = draw["env"], draw["episode"]
            edges = np.r_[0, np.flatnonzero(done[:, env]) + 1]
            start = int(edges[number - 1])
            if start == len(done):  # Reset at 60 s has no recorded samples.
                continue
            stop = int(edges[number]) if number < len(edges) else len(done)
            item = dict(environment=env, episode=number, start_s=start * .02, stop_s=stop * .02,
                        completed=bool(done[stop - 1, env]))
            if tag == "isaac400":
                assert start == draw["policy_step"]
                offsets = np.asarray(draw["com_offset_body_m"])
                # The shared draw is quantized differently when added to each float32 baseline CoM.
                assert np.max(abs(offsets - offsets[:1])) < 1e-7
                material = np.asarray(draw["material_static_dynamic_restitution"])
                item.update(body_order=draw["body_names"], material_order="PhysX collision-shape slots",
                            mass_scale=draw["mass_scale"], mass_kg=draw["mass_kg"],
                            com_offset_body_m=draw["com_offset_body_m"],
                            com_shared_offset_m=offsets[0].tolist(),
                            contact_static_friction=material[:, 0].tolist(),
                            contact_dynamic_friction=material[:, 1].tolist(),
                            restitution=material[:, 2].tolist(), kp=draw["kp"], kd=draw["kd"],
                            initial_roll_rad=roll_xyzw(draw["root_pose_xyzw"][3:7]),
                            initial_roll_rate_rad_s=draw["root_velocity_world"][3],
                            gyro_bias_rad_s=draw["gyro_bias_rad_s"], gravity_bias_yz=draw["gravity_bias_yz"],
                            source="Read-only Isaac repeat; all 18 original capture arrays byte-identical")
            else:
                dr = draw["dr"]
                w, x, y, z = draw["qpos"][3:7]
                item.update(body_order=["base_link", "rear_wheel_link", "steering_link", "front_wheel_link"],
                            material_order="Collision geometries in body_order",
                            mass_scale=dr["mass_scale"], com_shared_offset_m=dr["com_offset"],
                            com_offset_mujoco_body_m=dr["com_offset_mj_body"],
                            contact_static_friction=dr["tire_friction"],
                            contact_dynamic_friction=dr["tire_dynamic_friction"], restitution=[0.] * 4,
                            kp=draw["kp"], kd=draw["kd"],
                            gain_scales=[dr[key] for key in ("steer_kp_scale", "steer_kv_scale",
                                                            "drive_kp_scale", "drive_kv_scale")],
                            initial_roll_rad=roll_xyzw([x, y, z, w]), initial_roll_rate_rad_s=draw["qvel"][3],
                            gyro_bias_rad_s=dr["gyro_bias"], gravity_bias_yz=dr["accel_bias"],
                            source="Original read-only MuJoCo reset snapshots",
                            dynamic_friction_note=("Sampled dynamic coefficient retained; "
                                                   "native contact uses static only"))
            episodes.append(item)
        assert len(episodes) == int(done.sum()) + 64 - int(done[-1].sum())
        output["runs"][tag] = dict(snapshot_file=str(source.relative_to(HERE)),
                                  snapshot_sha256=digest(source), episodes=episodes)
    (HERE / "sampled_dr_values.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    counts = {tag: len(run["episodes"]) for tag, run in output["runs"].items()}
    print(json.dumps({"all_18_original_isaac_arrays_byte_identical": True, "plotted_episodes": counts}, indent=2))


if __name__ == "__main__":
    main()
