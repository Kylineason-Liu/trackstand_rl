#!/usr/bin/env python3
"""Viewer for the copied orange_bike model (tutorial 07). Loads the XML and
steps it passively -- zero actuator control -- so the raw physics can be
watched at realtime speed. Close the window to quit.

Usage:
    python3 scripts/run_orange_bike_viewer.py
"""

import os
import time

import mujoco
import mujoco.viewer  # not auto-imported by `import mujoco`

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models/orange_bike/orange_bike_horizontal.xml")


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("[INFO] viewer open -- zero control, close the window to quit")
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
