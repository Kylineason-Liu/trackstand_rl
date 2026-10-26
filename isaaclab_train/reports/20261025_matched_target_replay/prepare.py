"""Create immutable, SI-unit motor inputs from actual model_2500 DR captures."""

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = HERE.parent / "20261025_2500_dr_comparison"
DT = .02


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if (HERE / "inputs.npz").exists():
        raise FileExistsError("Preserve existing replay inputs; use a separate report for another experiment")
    targets, rear, names, sources = [], [], [], {}
    time = np.arange(250) * DT
    for name, signal in (("sine_1hz_0p1rad", .1 * np.sin(2 * np.pi * time)),
                         ("alternating_25hz_0p1rad", .1 * (-1.) ** np.arange(250))):
        names.append(name)
        targets.append(signal.astype(np.float32))
        rear.append(np.zeros(250, dtype=np.float32))
    starts = []
    for tag in ("isaac400", "mujoco400", "mujoco800"):
        path = SOURCE / tag / ("motor_samples.npz" if tag == "isaac400" else "physics_samples.npz")
        with np.load(path) as archive:
            target = archive["held_steering_target_rad"]
            torque = archive["submitted_motor_torque_nm"]
            targets.append(target[625:875, 0, 0].astype(np.float32))
            rear.append(torque[625:875, 0, 0, 1].astype(np.float32))
            starts.append((target[:400, 0, 0].astype(np.float32), torque[:400, 0, 0, 1].astype(np.float32)))
        names.append(f"{tag}_dr_env0_12p5_17p5")
        sources[str(path.relative_to(ROOT))] = sha(path)
    # 1s linear preparation +1s hold, 5s actual sequence, 1s final hold.
    prepared_target, prepared_rear = [], []
    for target, torque in zip(targets, rear):
        prefix = np.r_[np.linspace(0, target[0], 51)[1:], np.full(50, target[0])]
        prepared_target.append(np.r_[prefix, target, np.full(50, target[-1])].astype(np.float32))
        prepared_rear.append(np.r_[np.zeros(100), torque, np.full(50, torque[-1])].astype(np.float32))
    # Paired steering-friction on/off copies; rear friction stays unchanged.
    air_target = np.repeat(np.stack(prepared_target, axis=1), 2, axis=1)
    air_rear = np.repeat(np.stack(prepared_rear, axis=1), 2, axis=1)
    air_names = [f"{name}_{loss}" for name in names for loss in ("friction_on", "friction_off")]
    # Ground replay uses original commands FROM SPAWN, with no preparation.
    ground_target = np.stack([starts[0][0], starts[2][0]], axis=1)
    ground_rear = np.stack([starts[0][1], starts[2][1]], axis=1)
    assert np.max(abs(air_target)) <= 1.57 and np.max(abs(ground_target)) <= 1.57
    assert np.max(abs(np.diff(np.vstack([np.zeros((1, 10)), air_target]), axis=0))) <= .478 + 1e-6
    assert np.max(abs(np.diff(np.vstack([np.zeros((1, 2)), ground_target]), axis=0))) <= .478 + 1e-6
    assert max(np.max(abs(air_rear)), np.max(abs(ground_rear))) <= 6.28
    np.savez_compressed(HERE / "inputs.npz", policy_dt_s=DT, time_s=np.arange(400) * DT,
                        airborne_target_rad=air_target, airborne_rear_nm=air_rear,
                        airborne_names=np.asarray(air_names),
                        ground_target_rad=ground_target, ground_rear_nm=ground_rear,
                        ground_names=np.asarray(["isaac_dr_env0_from_spawn", "mujoco800_dr_env0_from_spawn"]))
    metadata = dict(source_sha256=sources, input_sha256=sha(HERE / "inputs.npz"), duration_s=8.,
                    core_window_s=[2., 7.], final_hold_window_s=[7., 8.], policy_hz=50,
                    supported_fixture="fixed base, z=2m, no wheel-ground contact; gravity retained",
                    grounded_fixture="original floating bike, nominal spawn, no automatic reset",
                    parameters="matched nominal v1 gains/masses/CoMs; no DR resampling or policy inference",
                    replay_sources="actual post-scale/post-clip/post-slew model_2500 DR commands",
                    precision="One shared float32 input file; MuJoCo original float64 targets rounded once <1e-7rad",
                    friction_off="steering static/dynamic/viscous losses zero; rear losses unchanged")
    (HERE / "inputs.json").write_text(json.dumps(metadata, indent=2) + "\n")
    protected = [ROOT / "progress_record.md", ROOT / "progress_record.txt", ROOT / "README.MD", ROOT / ".git/index"]
    for folder in (ROOT / "isaaclab_train/source", ROOT / "isaaclab_train/scripts",
                   ROOT / "mujoco_sim2sim/src", ROOT / "mujoco_sim2sim/scripts"):
        protected += [path for path in folder.rglob("*") if path.is_file() and path.suffix in (".py", ".yaml", ".usda")]
    protected += [ROOT / "mujoco_sim2sim/models/trackstand_rl_0916_bike/trackstand_rl_0916.xml"]
    protected += [path for path in SOURCE.rglob("*") if path.is_file()]
    (HERE / "protected_before.json").write_text(json.dumps({str(p): sha(p) for p in protected}, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
