"""Repeat the original capture with read-only, post-reset DR snapshots.

Use recovered draws for old figures ONLY after exact archive-array equality.
The original recorder, captures, task and evaluator are never rewritten.
"""

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def install_dr_recorder(recorder, output, tensor):
    env, robot = recorder.env, recorder.robot
    counts = [0] * recorder.n
    draws = []
    original_reset, original_save = env._reset_idx, recorder.save

    def reset(ids):
        original_reset(ids)
        indices = tensor(ids).cpu().tolist()
        mass_term = env.event_manager.get_term_cfg("randomize_mass").func
        com_term = env.event_manager.get_term_cfg("randomize_com").func
        masses = tensor(robot.root_view.get_masses()).cpu().numpy()
        default_mass = tensor(mass_term.default_mass).cpu().numpy()
        com = tensor(robot.data.body_com_pose_b).cpu().numpy()
        default_com = tensor(com_term.default_com).cpu().numpy()
        material = tensor(robot.root_view.get_material_properties()).cpu().numpy()
        pose = tensor(robot.data.root_pose_w).cpu().numpy()
        velocity = tensor(robot.data.root_vel_w).cpu().numpy()
        gyro = env._imu_gyro_bias.cpu().numpy()
        tilt = env._imu_tilt_bias.cpu().numpy()
        kp = [tensor(act.stiffness).cpu().numpy() for act in recorder.actuators]
        kd = [tensor(act.damping).cpu().numpy() for act in recorder.actuators]
        for index in indices:
            counts[index] += 1
            draws.append(dict(
                env=index, episode=counts[index],
                policy_step=recorder.completed + int(recorder.active),
                body_names=robot.body_names,
                mass_kg=masses[index].tolist(), mass_scale=(masses[index] / default_mass[index]).tolist(),
                com_offset_body_m=(com[index, :, :3] - default_com[index, :, :3]).tolist(),
                material_static_dynamic_restitution=material[index].tolist(),
                kp=[float(value[index, 0]) for value in kp], kd=[float(value[index, 0]) for value in kd],
                root_pose_xyzw=pose[index].tolist(), root_velocity_world=velocity[index].tolist(),
                gyro_bias_rad_s=gyro[index].tolist(), gravity_bias_yz=tilt[index].tolist()))

    def save():
        original_save()
        if recorder.saved:
            (output / "dr_draws.json").write_text(json.dumps(draws, indent=2, allow_nan=False) + "\n")

    env._reset_idx, recorder.save = reset, save


if __name__ == "__main__":
    driver = (HERE / "record_isaac.py").read_text()
    needle = 'exec(compile(source, str(TEMPLATE), "exec"), namespace)'
    assert driver.count(needle) == 1
    injected = (
        'source = source.replace("recorder = Recorder(env)", '
        '"recorder = Recorder(env)\\n    install_dr_recorder(recorder, OUT, tensor)")\n'
        'namespace["install_dr_recorder"] = install_dr_recorder\n' + needle)
    scope = {"__name__": "__recover_dr__", "__file__": str(HERE / "record_isaac.py"),
             "install_dr_recorder": install_dr_recorder}
    exec(compile(driver.replace(needle, injected), str(HERE / "record_isaac.py"), "exec"), scope)
