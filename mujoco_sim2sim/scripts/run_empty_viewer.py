#!/usr/bin/env python3
"""Open an EMPTY MuJoCo window: a lit, empty world with a floor. Close the
window or press Esc to quit.

Usage:
    python3 scripts/run_empty_viewer.py
"""

import time

import mujoco
import mujoco.viewer  # not auto-imported by `import mujoco`

EMPTY_XML = """
<mujoco>
  <worldbody>
    <light directional="true" pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="5 5 0.1"/>
  </worldbody>
</mujoco>
"""


def main():
    model = mujoco.MjModel.from_xml_string(EMPTY_XML)
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("[INFO] empty viewer open -- close the window to quit")
        while viewer.is_running():
            time.sleep(0.05)


if __name__ == "__main__":
    main()
