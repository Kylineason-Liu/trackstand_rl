#!/usr/bin/env python3
"""Free-fall test with the window, in the same minimal form as
run_orange_bike_viewer.py: load the model XML directly (model.xml now carries
its own ground and lighting), open a passive viewer, step at realtime speed.

The ACTUATORS ARE ALWAYS DISABLED: the gains/bias are zeroed, so the joints
are totally free and loose (without this, the steering position servo would
hold 0 and the drive damping would brake the wheel).

Prints the fall time, using the env's own fallen term (|gy| > sin30 deg or
inverted), measured from the free-joint qpos quat. The bike stays where it
lands. Close the window to quit.

Usage:
    python3 scripts/run_free_fall_viewer.py
"""

import os
import time

import mujoco
import mujoco.viewer  # not auto-imported by `import mujoco`
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#MODEL_PATH = os.path.join(ROOT, "models/trackstand_rl_bike/trackstand_rl.xml")
MODEL_PATH = os.path.join(ROOT, "models/trackstand_rl_0914_bike/trackstand_rl.xml")


FALLEN_GY = np.sin(np.deg2rad(30.0))


def _rot_matrix(q):
    """Rotation matrix from a w-first quaternion (same formula as env._rot_matrix)."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    # the FREE state: the actuators fully inert (totally free and loose)
    for i in range(model.nu):
        model.actuator_gainprm[i, :] = 0.0
        model.actuator_biasprm[i, :] = 0.0
    print("[INFO] actuators disabled -- the bike falls freely (close the window to quit)")

    fallen_at = None
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("[INFO] viewer open -- close the window to quit")
        while viewer.is_running():
            # the "Convex Hull" button toggles mjVIS_CONVEXHULL in the shared
            # MjvOption; mirror it into geomgroup[3] so the button also
            # shows/hides the collision geoms (group 3, hidden by default)
            viewer.opt.geomgroup[3] = viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL]
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            if fallen_at is None:
                # the free joint is the first joint: qpos[3:7] = base quat (w-first)
                R = _rot_matrix(data.qpos[3:7])
                g_b = R.T @ np.array([0.0, 0.0, -1.0])
                if abs(g_b[1]) > FALLEN_GY or g_b[2] > 0.0:
                    fallen_at = data.time
                    print(f"FALL TIME (mujoco, free joints): {fallen_at:.2f} s  "
                          f"({fallen_at / 0.02:.0f} steps)")
    if fallen_at is None:
        fallen_at = data.time
        print(f"FALL TIME (mujoco, free joints): {fallen_at:.2f} s  ({fallen_at / 0.02:.0f} steps)")


if __name__ == "__main__":
    main()

